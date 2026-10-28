"""LLM-backed variable renamer (v3): a sibling of rename_vars.py.

    from rename_vars_llm import LocalLLM, rename_vars_llm
    llm = LocalLLM(model='qwen2.5-coder', base_url='http://localhost:8080/v1',
                   cache_path='names.cache.jsonl')
    rename_vars_llm(session, llm, max_prompt_chars=60000)     # for -c 32768

The model only *proposes* names. This module decides whether a name is usable:
eligibility, type inference and the shorthand fix come from rename_vars.py, and
every name is checked for capture before it is written into the tree.

Strategy
--------
Work is split into *items*, each one request-sized view of the code:

* CLUSTER. A function whose whole subtree fits `cluster_chars` becomes one item
  with all its nested functions visible, so the model sees a runtime/closure as
  a whole instead of fifteen isolated fragments. Every function in view carries
  a /* F12 */ marker that the reply uses as its key.
* SINGLETON. A bigger function is its own item; big children are collapsed to
  `header + /* purpose */`, and any use of the asked-about variables *inside* a
  collapsed child is shown as an excerpt in that stub, so a variable that is
  only used in closures is not invisible. If it still does not fit, only the
  statements that touch the asked-about variables are sent.
* ORDER. Items run callees-first (nested functions, then functions referenced),
  so a caller sees helpers that already have names and purposes.
* OVERVIEW. If the file does not fit in one cluster, one request first sees a
  skeleton (bodies collapsed to ids + a fingerprint of strings and properties)
  and returns a description of the file plus guesses for the functions it can
  recognise. The description heads every later request; guesses are hints.
* CLONES. Functions with an identical structure (names and number values
  ignored) are named once and the names are copied to the other copies.
* REPAIR. Names that had to be suffixed because they collided get one
  follow-up asking for a more specific name, instead of `thing2`.
"""
import hashlib
import heapq
import json
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Set, Tuple

import escodegen

from jsir import Session, walk, to_esprima
from jsir import values as V
from jsir.schema import FUNCTION_KINDS, slots_of, VALUE, KEY, LABEL, attributes_of
from jsir.tree import Node


# ---------------------------------------------------------------------------
# Type inference
# ---------------------------------------------------------------------------

ARRAY_METHODS = frozenset((
    'push', 'pop', 'shift', 'unshift', 'splice', 'join', 'map', 'filter',
    'forEach', 'reduce', 'reduceRight', 'some', 'every', 'find', 'findIndex',
    'sort', 'reverse', 'fill', 'flat', 'flatMap'))
STRING_METHODS = frozenset((
    'charAt', 'charCodeAt', 'codePointAt', 'substr', 'substring', 'split',
    'replace', 'replaceAll', 'toLowerCase', 'toUpperCase', 'trim', 'trimStart',
    'trimEnd', 'padStart', 'padEnd', 'startsWith', 'endsWith', 'match',
    'matchAll', 'normalize', 'localeCompare', 'repeat'))
FUNCTION_METHODS = frozenset(('call', 'apply', 'bind'))

NUMERIC_BINARY = frozenset(('-', '*', '/', '%', '**', '&', '|', '^', '<<', '>>', '>>>'))
COMPARE_BINARY = frozenset(('===', '!==', '==', '!=', '<', '<=', '>', '>='))
BOOLEAN_BINARY = frozenset(('===', '!==', '==', '!=', '<', '<=', '>', '>=', 'instanceof', 'in'))
NUMERIC_ASSIGN = frozenset(('-=', '*=', '/=', '%=', '**=', '&=', '|=', '^=', '<<=', '>>=', '>>>='))

GLOBAL_ARGS = {name: 'str' for name in (
    'atob', 'btoa', 'escape', 'unescape', 'encodeURIComponent', 'encodeURI',
    'decodeURIComponent', 'decodeURI', 'parseFloat')}
STATIC_ARGS = {
    ('String', 'fromCharCode'): 'num', ('String', 'fromCodePoint'): 'num',
    ('JSON', 'parse'): 'str',
}

GLOBAL_CALLS = {
    'String': 'str', 'Number': 'num', 'parseInt': 'num', 'parseFloat': 'num',
    'Boolean': 'bool', 'isNaN': 'bool', 'isFinite': 'bool', 'RegExp': 're',
    'Array': 'arr', 'Object': 'obj', 'encodeURIComponent': 'str',
    'decodeURIComponent': 'str', 'encodeURI': 'str', 'decodeURI': 'str',
    'escape': 'str', 'unescape': 'str', 'atob': 'str', 'btoa': 'str',
}
STATIC_CALLS = {
    ('Array', 'isArray'): 'bool', ('Array', 'from'): 'arr', ('Array', 'of'): 'arr',
    ('Object', 'keys'): 'arr', ('Object', 'values'): 'arr', ('Object', 'entries'): 'arr',
    ('Object', 'assign'): 'obj', ('Object', 'create'): 'obj',
    ('JSON', 'stringify'): 'str', ('String', 'fromCharCode'): 'str',
    ('String', 'fromCodePoint'): 'str', ('Date', 'now'): 'num',
}
METHOD_RETURNS = {
    **{m: 'str' for m in ('join', 'toString', 'toFixed', 'charAt', 'substr', 'substring',
                          'toLowerCase', 'toUpperCase', 'trim', 'trimStart', 'trimEnd',
                          'padStart', 'padEnd', 'repeat', 'replace', 'replaceAll', 'normalize')},
    **{m: 'num' for m in ('charCodeAt', 'codePointAt', 'indexOf', 'lastIndexOf', 'push',
                          'unshift', 'localeCompare')},
    **{m: 'arr' for m in ('split', 'map', 'filter', 'splice', 'flat', 'flatMap',
                          'reverse', 'sort', 'exec', 'match')},
    **{m: 'bool' for m in ('includes', 'startsWith', 'endsWith', 'test', 'some', 'every',
                           'hasOwnProperty')},
}

FIXED_KINDS = {'catch': 'err', 'function': 'fn', 'class': 'cls'}


def _unify(a: Optional[str], b: Optional[str]) -> Optional[str]:
    if a == b:
        return a
    if a is None:
        return b
    if b is None:
        return a
    return None


def _prop_name(member) -> Optional[str]:
    """Name of a non-computed member property, else None."""
    if member.attr('computed', False):
        return None
    prop = member.get('property')
    return prop.attr('name') if prop is not None and prop.kind == 'Identifier' else None


class TypeInferer:
    """Coarse type inference: 'num','str','bool','arr','obj','fn','cls','re',
    'err' or None for unknown. Combines declaration/assignment evidence
    (weight 2) with usage evidence (weight 1) and follows identifier flow."""

    def __init__(self, session: Session):
        self.session = session
        self.scopes = session.scopes
        self._memo: Dict[object, Optional[str]] = {}
        self._active: Set[object] = set()

    # -- bindings ----------------------------------------------------------

    def binding(self, binding) -> Optional[str]:
        if binding in self._memo:
            return self._memo[binding]
        if binding in self._active:          # cycle: a = b; b = a
            return None
        self._active.add(binding)
        try:
            result = self._resolve(binding)
        finally:
            self._active.discard(binding)
        self._memo[binding] = result
        return result

    def _resolve(self, binding) -> Optional[str]:
        fixed = FIXED_KINDS.get(binding.kind)
        if fixed:
            return fixed

        votes: Counter = Counter()
        target = binding.target
        decl = binding.declaration

        def vote(type_, weight):
            if type_:
                votes[type_] += weight

        if decl is not None and decl.kind == 'VariableDeclarator':
            if decl.get('id') is target:                       # not a destructuring pattern
                vote(self.expr(decl.get('init')), 2)
                if decl.get('init') is None:
                    loop = decl.parent.parent if decl.parent is not None else None
                    if (loop is not None and loop.kind == 'ForInStatement'
                            and decl.parent.parent_slot == 'left'):
                        vote('str', 2)

        if target is not None and target.parent is not None:
            p = target.parent
            if p.kind == 'AssignmentPattern' and p.get('left') is target:
                vote(self.expr(p.get('right')), 2)             # default parameter value

        for w in binding.writes:
            if w is not target:
                vote(self.write(w), 2)
        for r in binding.reads:
            vote(self.read(r), 1)

        if not votes:
            return None
        ranked = votes.most_common(2)
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            return None
        return ranked[0][0]

    # -- evidence from writes ---------------------------------------------

    def write(self, ident) -> Optional[str]:
        parent = ident.parent
        if parent is None:
            return None
        kind = parent.kind
        if kind == 'AssignmentExpression' and parent.get('left') is ident:
            op = parent.attr('operator', '=')
            right = parent.get('right')
            if op in ('=', '||=', '&&=', '??='):
                return self.expr(right)
            if op == '+=':
                # `s += 1` says nothing about s; only a string operand is conclusive.
                return 'str' if self.expr(right) == 'str' else None
            if op in NUMERIC_ASSIGN:
                return 'num'
        if kind == 'UpdateExpression':
            return 'num'
        if kind == 'ForInStatement' and parent.get('left') is ident:
            return 'str'
        return None

    # -- evidence from reads ----------------------------------------------

    def read(self, ident) -> Optional[str]:
        parent = ident.parent
        if parent is None:
            return None
        kind, slot = parent.kind, ident.parent_slot
        if kind == 'MemberExpression' and slot == 'object':
            name = _prop_name(parent)
            if name is None:
                return None
            if name in ARRAY_METHODS:
                return 'arr'
            if name in STRING_METHODS:
                return 'str'
            if name in FUNCTION_METHODS:
                return 'fn'
            if name == 'length':
                return None                    # array or string
            return 'obj'
        if kind == 'CallExpression' and slot == 'callee':
            return 'fn'
        if kind == 'NewExpression' and slot == 'callee':
            return 'cls'
        if kind == 'BinaryExpression':
            op = parent.attr('operator')
            if op in NUMERIC_BINARY:
                return 'num'
            if op in COMPARE_BINARY:
                # `x === 'abc'`, `x < 3`: x shares the type of the other side.
                other = parent.get('right') if slot == 'left' else parent.get('left')
                seen = self.expr(other)
                return seen if seen in ('num', 'str', 'bool') else None
        if kind == 'CallExpression' and slot == 'arguments':
            return self._argument_type(parent)
        if kind == 'UpdateExpression':
            return 'num'
        if kind == 'UnaryExpression' and parent.attr('operator') in ('-', '+', '~'):
            return 'num'
        if kind == 'ForInStatement' and slot == 'right':
            return 'obj'
        if kind == 'ForOfStatement' and slot == 'right':
            return 'arr'
        return None

    # -- expressions ------------------------------------------------------

    def expr(self, node) -> Optional[str]:
        if node is None:
            return None
        session = self.session

        c = session.constant_of(node)
        if c is not None and c not in (V.UNSET, V.UNKNOWN):
            if isinstance(c, bool):
                return 'bool'
            if isinstance(c, (int, float)):
                return 'num'
            if isinstance(c, str):
                return 'str'
            if isinstance(c, V.JSRegExp):
                return 're'

        kind = node.kind

        if kind == 'Literal':
            val = node.attr('value')
            if val is V.NULL:
                return None
            if isinstance(val, bool):
                return 'bool'
            if isinstance(val, (int, float)):
                return 'num'
            if isinstance(val, str):
                return 'str'
            if isinstance(val, V.JSRegExp):
                return 're'
            return None

        if kind == 'TemplateLiteral':
            return 'str'

        if kind == 'Identifier':
            name = node.attr('name')
            if name == 'undefined':
                return None
            b = self.scopes.binding_of(node)
            if b is None:
                return 'num' if name in ('NaN', 'Infinity') else None
            return self.binding(b)

        if kind in FUNCTION_KINDS:
            return 'fn'
        if kind in ('ClassExpression', 'ClassDeclaration'):
            return 'cls'
        if kind == 'ArrayExpression':
            return 'arr'
        if kind == 'ObjectExpression':
            return 'obj'

        if kind == 'UnaryExpression':
            op = node.attr('operator')
            if op in ('!', 'delete'):
                return 'bool'
            if op == 'typeof':
                return 'str'
            if op in ('+', '-', '~'):
                return 'num'
            return None

        if kind == 'UpdateExpression':
            return 'num'

        if kind == 'BinaryExpression':
            op = node.attr('operator')
            if op in BOOLEAN_BINARY:
                return 'bool'
            if op in NUMERIC_BINARY:
                return 'num'
            if op == '+':
                lt, rt = self.expr(node.get('left')), self.expr(node.get('right'))
                if lt == 'str' or rt == 'str':
                    return 'str'
                if lt == 'num' and rt == 'num':
                    return 'num'
            return None

        if kind in ('ConditionalExpression',):
            return _unify(self.expr(node.get('consequent')), self.expr(node.get('alternate')))

        if kind == 'LogicalExpression':
            return _unify(self.expr(node.get('left')), self.expr(node.get('right')))

        if kind == 'AssignmentExpression':
            op = node.attr('operator', '=')
            if op == '=':
                return self.expr(node.get('right'))
            return None

        if kind == 'SequenceExpression':
            exprs = node.children_in('expressions')
            return self.expr(exprs[-1]) if exprs else None

        if kind == 'ChainExpression':
            return self.expr(node.get('expression'))

        if kind == 'MemberExpression':
            return 'num' if _prop_name(node) == 'length' else None

        if kind == 'CallExpression':
            return self._call(node)

        if kind == 'NewExpression':
            callee = node.get('callee')
            if callee is not None and callee.kind == 'Identifier':
                name = callee.attr('name')
                if name == 'RegExp':
                    return 're'
                if name == 'Array':
                    return 'arr'
            return 'obj'

        return None

    def _argument_type(self, call) -> Optional[str]:
        """What a well-known function implies about the type of its arguments."""
        callee = call.get('callee')
        if callee is None:
            return None
        if callee.kind == 'Identifier':
            if self.scopes.binding_of(callee) is None:
                return GLOBAL_ARGS.get(callee.attr('name'))
            return None
        if callee.kind == 'MemberExpression':
            obj = callee.get('object')
            if obj is not None and obj.kind == 'Identifier' and \
                    self.scopes.binding_of(obj) is None:
                owner = obj.attr('name')
                if owner == 'Math':
                    return 'num'
                return STATIC_ARGS.get((owner, _prop_name(callee)))
        return None

    def _call(self, node) -> Optional[str]:
        callee = node.get('callee')
        if callee is None:
            return None
        if callee.kind == 'Identifier':
            if self.scopes.binding_of(callee) is None:
                return GLOBAL_CALLS.get(callee.attr('name'))
            return None
        if callee.kind == 'MemberExpression':
            name = _prop_name(callee)
            obj = callee.get('object')
            if obj is not None and obj.kind == 'Identifier' and \
                    self.scopes.binding_of(obj) is None:
                owner = obj.attr('name')
                if owner == 'Math':
                    return 'num'
                hit = STATIC_CALLS.get((owner, name))
                if hit:
                    return hit
            return METHOD_RETURNS.get(name)
        return None


# ---------------------------------------------------------------------------
# Scope analysis
# ---------------------------------------------------------------------------

def _is_reference_position(ident) -> bool:
    """False for labels and non-computed property keys, which are not variables."""
    parent = ident.parent
    if parent is None:
        return False
    role = next((r for name, _a, r in slots_of(parent.kind) if name == ident.parent_slot), VALUE)
    if role == LABEL:
        return False
    if role == KEY and not parent.attr('computed', False):
        return False
    return True


def _collect_free_names(session: Session, targets: Set[object]) -> Set[str]:
    """Names that must never be generated: session globals plus every
    identifier that is a real reference with no local declaration."""
    reserved: Set[str] = set()

    for attr in ('runtime_object', 'original_object'):
        value = getattr(session, attr, None)
        if value:
            reserved.add(value)
    if hasattr(session, 'oracle') and hasattr(session.oracle, 'global_aliases'):
        reserved.update(session.oracle.global_aliases)

    scopes = session.scopes
    for ident in walk(session.tree.root, kinds=('Identifier',)):
        if ident in targets or scopes.binding_of(ident) is not None:
            continue
        if not _is_reference_position(ident):
            continue
        name = ident.attr('name')
        if name:
            reserved.add(name)
    return reserved


def _scopes_top_down(root) -> list:
    order, stack = [], [root]
    while stack:
        scope = stack.pop()
        order.append(scope)
        stack.extend(reversed(scope.children))
    return order


def _dynamic_exposed_scopes(all_scopes) -> Set[object]:
    """Scopes whose bindings code inside eval/with could reach by name: the
    dynamic scope itself, its descendants, and every ancestor."""
    exposed: Set[object] = set()
    for scope in all_scopes:
        if scope.has_dynamic_scope:
            cur = scope
            while cur is not None and cur not in exposed:
                exposed.add(cur)
                cur = cur.parent
    return exposed


def _is_exported(binding) -> bool:
    node = binding.declaration
    for _ in range(3):
        if node is None:
            return False
        if node.kind in ('ExportNamedDeclaration', 'ExportDefaultDeclaration'):
            return True
        node = node.parent
    return False


def _fix_shorthand_properties(root) -> None:
    """`{a}` shorthand reads its key from the value. After the value is renamed
    the key must stay as written, so expand to `{a: x}`."""
    for prop in walk(root, kinds=('Property',)):
        if not prop.attr('shorthand', False):
            continue
        key, value = prop.get('key'), prop.get('value')
        if value is not None and value.kind == 'AssignmentPattern':
            value = value.get('left')
        if key is None or value is None or key.kind != 'Identifier' or value.kind != 'Identifier':
            continue
        if key.attr('name') != value.attr('name'):
            prop.attrs['shorthand'] = False


# ---------------------------------------------------------------------------
# Local model client
# ---------------------------------------------------------------------------


class LLMError(RuntimeError):
    """The server could not be reached or answered with an error."""


class LocalLLM:
    """Client for an OpenAI-compatible /chat/completions endpoint (llama-server,
    Ollama, LM Studio, vLLM). Safe to call from several threads.

    :param cache_path: JSONL file of past replies keyed by the exact request.
        Reruns and resumes skip every request already answered. Replies that
        could not be used are not stored.
    :param extra_body: merged into the request, e.g. {"cache_prompt": True}.
    """

    def __init__(self, model: str, base_url: str = 'http://localhost:11434/v1',
                 api_key: Optional[str] = None, temperature: float = 0.0,
                 max_tokens: int = 1024, timeout: float = 900.0,
                 json_mode: bool = False, extra_body: Optional[dict] = None,
                 cache_path: Optional[str] = None):
        self.model = model
        self.url = base_url.rstrip('/') + '/chat/completions'
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.json_mode = json_mode
        self.extra_body = extra_body or {}
        self.cache_path = cache_path
        self.cache: Dict[str, str] = {}
        self.calls = 0
        self.cache_hits = 0
        self._lock = threading.Lock()
        if cache_path:
            try:
                with open(cache_path, encoding='utf-8') as handle:
                    for line in handle:
                        row = json.loads(line)
                        self.cache[row['k']] = row['v']
            except FileNotFoundError:
                pass

    def complete(self, system: str, user: str, attempt: int = 0,
                 max_tokens: Optional[int] = None, validate=None) -> str:
        return self.complete_info(system, user, attempt, max_tokens, validate)[0]

    def complete_info(self, system: str, user: str, attempt: int = 0,
                      max_tokens: Optional[int] = None, validate=None):
        """-> (text, from_cache, seconds). attempt > 0 (a retry after an unusable
        reply) raises the temperature; repeating a temperature-0 request would
        return the same bad text. `validate(text) -> bool` gates what is
        written to the cache."""
        body = {
            'model': self.model,
            'messages': [{'role': 'system', 'content': system},
                         {'role': 'user', 'content': user}],
            'temperature': max(self.temperature, min(1.0, 0.3 * attempt)),
            'max_tokens': max_tokens or self.max_tokens,
            'stream': False,
            **self.extra_body,
        }
        if self.json_mode:
            body['response_format'] = {'type': 'json_object'}
        key = hashlib.sha1(json.dumps(body, sort_keys=True).encode()).hexdigest()
        with self._lock:
            if key in self.cache:
                self.cache_hits += 1
                return self.cache[key], True, 0.0

        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        request = urllib.request.Request(self.url, json.dumps(body).encode(), headers, method='POST')
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.load(response)
        except urllib.error.HTTPError as e:
            raise LLMError(f'{self.url} -> HTTP {e.code}: {e.read()[:300]!r}') from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise LLMError(f'cannot reach {self.url}: {e}') from e
        try:
            text = data['choices'][0]['message']['content'] or ''
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f'unexpected response shape: {str(data)[:300]}') from e
        seconds = time.monotonic() - started

        with self._lock:
            self.calls += 1
            if validate is None or validate(text):
                self.cache[key] = text
                if self.cache_path:
                    with open(self.cache_path, 'a', encoding='utf-8') as handle:
                        handle.write(json.dumps({'k': key, 'v': text}) + '\n')
        return text, False, seconds


# ---------------------------------------------------------------------------
# Prompts, reply parsing, name hygiene
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a JavaScript reverse-engineering assistant. You rename variables in minified or obfuscated code so it reads like hand-written code.

The message may start with a "File:" line describing the whole file. Then come numbered blocks: some code, and a list of items to name grouped by function id such as [F3]. Functions in the code are marked with a comment like /* F3 */ at the start of their body. Reply with ONE JSON object, no prose: its keys are the function ids, and each value maps every listed item to your answer.

Items:
- "@name": a name for the function itself (a verb phrase such as parseHeader).
- "@purpose": one plain sentence (max 15 words) saying what the function does.
- anything else is a variable: map its old name to a new name.

Rules for names:
- Say what the variable holds or what role it plays, judging from how the code uses it, including uses inside nested functions.
- Be specific. Avoid vague names such as data, obj, item, value, temp, thing, unless nothing more specific can be known.
- Different variables get different names; if two things are similar, say what distinguishes them.
- camelCase; PascalCase only for classes. A valid JavaScript identifier, at most 40 characters, not a reserved word.
- Conventional short names (i, j, e, err, cb) are fine for loop counters, caught errors and callbacks.
- Names already in the code were chosen earlier; use them as hints. A comment such as /* ... */ in place of a body summarises that function.
- If you cannot tell what something is for, use a plain type-based name (str, list, fn).

Example reply:
{"F3": {"@name": "parseHeader", "@purpose": "Splits a raw HTTP header block into a key/value map.", "a": "rawHeaders", "b": "headerMap"}, "F4": {"@purpose": "Reads one header line.", "e": "line"}}"""

OVERVIEW_PROMPT = """\
You are a JavaScript reverse-engineering assistant. You are shown the structure of a large minified or obfuscated JavaScript file: top-level code and some functions in full, with other function bodies collapsed to a comment holding an id, a node count, and a few distinctive strings and property names found inside.

Reply with ONE JSON object, no prose:
- "@file": 2-3 sentences: what kind of code this is (bundler runtime, framework, library, application), what it appears to do, and how it is organised.
- "F12": {"@name": "...", "@purpose": "..."} for each function whose id you can identify with fair confidence. Skip any you are unsure about. At most 40 entries. "@name" is a verb phrase in camelCase, "@purpose" one short sentence."""

MAX_NAME_LEN = 48

JS_RESERVED = frozenset("""
break case catch class const continue debugger default delete do else enum export
extends false finally for function if import in instanceof new null return super
switch this throw true try typeof var void while with yield let static implements
interface package private protected public await async of get set
arguments eval undefined NaN Infinity
""".split())

TYPE_STEMS = {'num': 'num', 'str': 'str', 'arr': 'list', 'obj': 'obj', 'bool': 'flag',
              'fn': 'fn', 'cls': 'Cls', 're': 'rx', 'err': 'err', 'gen': 'val'}

_THINK = re.compile(r'<think>.*?</think>', re.S | re.I)
_FENCE = re.compile(r'```(?:json)?\s*(.*?)```', re.S | re.I)


def parse_json_object(text: str) -> dict:
    """The first JSON object in a model reply (tolerates <think> blocks, fences,
    chatter). {} when there is none."""
    text = _THINK.sub('', text).strip()
    candidates = [text]
    fenced = _FENCE.search(text)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    first, last = text.find('{'), text.rfind('}')
    if 0 <= first < last:
        candidates.append(text[first:last + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(data, dict):
            return data
    return {}


def sanitize_identifier(raw) -> Optional[str]:
    """Coerce a proposed name into a legal, non-reserved identifier."""
    if not isinstance(raw, str):
        return None
    name = raw.strip().strip('`"\'')
    name = re.sub(r'[\s\-.]+', '_', name)
    name = re.sub(r'[^A-Za-z0-9_$]', '', name)
    if not name:
        return None
    if name[0].isdigit():
        name = '_' + name
    name = _cap_name(name)
    return name + '_' if name in JS_RESERVED else name


def _cap_name(name: str) -> str:
    """Shorten an over-long name at a word boundary, never mid-word."""
    if len(name) <= MAX_NAME_LEN:
        return name
    cut = name[:MAX_NAME_LEN]
    bounds = [m.start() for m in re.finditer(r'(?<=[a-z0-9])(?=[A-Z])|_', cut)]
    return cut[:bounds[-1]] if bounds and bounds[-1] >= 12 else cut


def is_echo(proposed: str, old: str) -> bool:
    """The model handed back the minified name instead of renaming it."""
    return proposed.lower() == old.lower() and (len(old) <= 3 or any(c.isdigit() for c in old))


def sanitize_text(raw, cap: int = 140) -> Optional[str]:
    """One line of prose that is safe inside a /* */ comment."""
    if not isinstance(raw, str):
        return None
    text = re.sub(r'\s+', ' ', raw.replace('*/', '').replace('/*', '')).strip()
    return (text[:cap].rstrip() + '...') if len(text) > cap else (text or None)


# ---------------------------------------------------------------------------
# Collision-free names
# ---------------------------------------------------------------------------

class NameTable:
    """Exact capture check.

    After renaming, every reference must still resolve to the binding it
    resolved to before. Giving binding X (in scope S) the name N breaks that
    only if
      1. S already has another binding named N, or
      2. a reference to X sits inside a scope below S that has its own N
         (that scope would now shadow X), or
      3. a binding Z named N in a scope above S has a reference inside S's
         subtree (S would now shadow Z).
    Each is checked directly from the recorded reference scopes, so sibling or
    unrelated scopes reuse names freely and a deep-but-unrelated binding no
    longer forces a numeric suffix. Names that are reserved (free globals,
    pinned bindings, keywords) are never used.
    """

    def __init__(self, reserved: Set[str], all_scopes, scope_of, nodes_of_binding,
                 unique_across_file: bool = False):
        self.reserved = reserved
        self.unique_across_file = unique_across_file
        self._scope_of = scope_of
        self._nodes_of = nodes_of_binding
        self.names: Dict[object, Dict[str, object]] = {s: dict(s.bindings) for s in all_scopes}
        self.current: Dict[object, str] = {}
        self.everywhere: Counter = Counter()
        for s in all_scopes:
            self.everywhere.update(list(s.bindings))
        self._refscopes: Dict[object, Set[object]] = {}

    def name_of(self, binding) -> str:
        return self.current.get(binding, binding.name)

    def refscopes(self, binding) -> Set[object]:
        got = self._refscopes.get(binding)
        if got is None:
            got = {self._scope_of(n) for n in self._nodes_of(binding)}
            got.discard(None)
            self._refscopes[binding] = got
        return got

    def _conflicts(self, scope, binding, name: str) -> bool:
        if name in self.reserved or name in JS_RESERVED:
            return True
        cur_name = self.name_of(binding)
        if self.unique_across_file and self.everywhere[name] - (1 if name == cur_name else 0) > 0:
            return True
        other = self.names[scope].get(name)
        if other is not None and other is not binding:                       # 1
            return True
        for s in self.refscopes(binding):                                    # 2
            cur = s
            while cur is not None and cur is not scope:
                hit = self.names[cur].get(name)
                if hit is not None and hit is not binding:
                    return True
                cur = cur.parent
        a = scope.parent                                                     # 3
        while a is not None:
            z = self.names[a].get(name)
            if z is not None and z is not binding:
                for s in self.refscopes(z):
                    cur = s
                    while cur is not None and cur is not a:
                        if cur is scope:
                            return True
                        cur = cur.parent
            a = a.parent
        return False

    def claim(self, scope, binding, stem: str, numbered: bool = False) -> str:
        n = 1 if numbered else 0
        while True:
            name = (stem + ('_' if stem[-1:].isdigit() else '') + str(n)) if n else stem
            if not self._conflicts(scope, binding, name):
                break
            n = max(n, 1) + 1
        old = self.name_of(binding)
        if name != old:
            if self.names[scope].get(old) is binding:
                del self.names[scope][old]
            self.names[scope][name] = binding
            self.everywhere[old] -= 1
            self.everywhere[name] += 1
            self.current[binding] = name
        return name

    def nearby(self, scope, limit: int = 40) -> List[str]:
        out, cur = [], scope
        while cur is not None and len(out) < limit:
            out.extend(n for n in self.names[cur] if n != 'arguments')
            cur = cur.parent
        return out[:limit]


# ---------------------------------------------------------------------------
# Units (functions), items (request-sized groups), jobs (chunks of an item)
# ---------------------------------------------------------------------------

class Unit:
    def __init__(self, fn):
        self.fn = fn                       # function node, or None for Program
        self.parent: Optional['Unit'] = None
        self.kids: List['Unit'] = []
        self.index = 0
        self.id = 'F0'
        self.size = 1                      # units in this subtree
        self.depth = 0
        self.own = 0                       # AST nodes belonging to this function only
        self.total = 0                     # ... including nested functions
        self.scopes: List[object] = []
        self.variables: List[Tuple[object, object]] = []
        self.by_name: Dict[str, List[Tuple[object, object]]] = {}
        self.fn_binding = None
        self.name_binding: Optional[Tuple[object, object]] = None
        self.deps: Dict['Unit', None] = {}
        self.purpose: Optional[str] = None
        self.hint_name: Optional[str] = None
        self.hint_purpose: Optional[str] = None
        self.item: Optional['Item'] = None
        self.aliased = False
        self.copied = False

    @property
    def end(self):
        return self.index + self.size

    def __repr__(self):
        return f'<Unit {self.id} own={self.own} total={self.total}>'


class Item:
    def __init__(self, root: Unit, members: List[Unit]):
        self.root = root
        self.members = members
        self.asks: List[Tuple[Unit, str]] = []
        self.deps: Dict['Item', None] = {}
        self.done = False

    @property
    def is_cluster(self):
        return len(self.members) > 1 or self.members[0].fn is None


class Job:
    def __init__(self, item: Item, asks, last: bool):
        self.item = item
        self.asks = asks
        self.last = last
        self.level = 0
        self.view = ''
        self.text = ''


def _owner_fn(node):
    cur = node.parent
    while cur is not None and cur.kind not in FUNCTION_KINDS:
        cur = cur.parent
    return cur


def _stmt_in(node, fn):
    """Smallest non-block statement around `node` within the code of `fn`.
    Climbs out of functions nested between `node` and `fn`."""
    best, cur = None, node
    while cur is not None and cur is not fn:
        if cur.kind in FUNCTION_KINDS and cur is not node:
            best = cur if cur.is_statement else None     # a function declaration is its own statement
        elif best is None and cur.is_statement and cur.kind != 'BlockStatement':
            best = cur
        cur = cur.parent
    return best


def _child_functions(node) -> List[Node]:
    """Functions directly nested in `node`'s own code, in source order."""
    out, stack = [], [c for _s, c, _r in reversed(list(node.iter_children()))]
    while stack:
        n = stack.pop()
        if n.kind in FUNCTION_KINDS:
            out.append(n)
        else:
            stack.extend(c for _s, c, _r in reversed(list(n.iter_children())))
    return out


def _dependency_order(nodes, deps_of):
    """Post-order DFS: everything comes after what it depends on, except across
    cycles (mutual recursion), where document order decides."""
    order, seen = [], set()
    for start in nodes:
        if start in seen:
            continue
        seen.add(start)
        stack = [(start, iter(list(deps_of(start))))]
        while stack:
            cur, it = stack[-1]
            for dep in it:
                if dep not in seen:
                    seen.add(dep)
                    stack.append((dep, iter(list(deps_of(dep)))))
                    break
            else:
                stack.pop()
                order.append(cur)
    return order


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_STUB = re.compile(r'__stub(\d+)__;?')
_FORMAT = {'format': {'indent': {'style': '  '}}}


def _sentinel(k: int) -> Node:
    ident = Node('Identifier', {'name': f'__stub{k}__'}, {})
    return Node('ExpressionStatement', {}, {'expression': ident})


def _clip(text: str, cap: int) -> str:
    return text if len(text) <= cap else text[:cap].rstrip() + ' /* ... */'


class Renderer:
    def __init__(self, session, units: Dict[object, Unit], collapse_nodes: int):
        self.session = session
        self.units = units
        self.collapse_nodes = collapse_nodes

    # -- primitive: print a node with some bodies swapped out -------------------
    def render(self, node, stubs=None, excerpts=None, markers=None) -> str:
        """Print `node`. `stubs` {fn: comment} replace a body by the comment
        (plus any `excerpts[fn]` statements); `markers` {fn: comment} prepend a
        comment to a body that is otherwise shown in full."""
        stubs, excerpts, markers = stubs or {}, excerpts or {}, markers or {}
        texts: List[str] = []
        undo = []

        def sentinel(text):
            texts.append(text)
            return _sentinel(len(texts) - 1)

        try:
            for fn, text in stubs.items():
                undo.append((fn, 'set', fn.children.get('body')))
                fn.children['body'] = Node('BlockStatement', {},
                                           {'body': [sentinel(text), *excerpts.get(fn, ())]})
            for fn, text in markers.items():
                body = fn.children.get('body')
                if fn in stubs or body is None:
                    continue
                if body.kind == 'BlockStatement':
                    lst = body.children['body']
                    lst.insert(0, sentinel(text))
                    undo.append((fn, 'ins', lst))
                else:                                      # expression-bodied arrow
                    undo.append((fn, 'set', body))
                    ret = Node('ReturnStatement', {}, {'argument': body})
                    fn.children['body'] = Node('BlockStatement', {}, {'body': [sentinel(text), ret]})
            code = escodegen.generate(to_esprima(node), _FORMAT)
        finally:
            for fn, how, val in reversed(undo):
                if how == 'set':
                    fn.children['body'] = val
                else:
                    val.pop(0)
        return _STUB.sub(lambda m: texts[int(m.group(1))], code)

    def stub_text(self, fn) -> str:
        unit = self.units[fn]
        return f'/* {unit.purpose} */' if unit.purpose else '/* ... */'

    def collapsible(self, node) -> Dict[Node, str]:
        return {fn: self.stub_text(fn) for fn in _child_functions(node)
                if self.units[fn].total >= self.collapse_nodes}

    # -- closure visibility -----------------------------------------------------
    def plan(self, node, wanted: Set[Node], depth: int = 0):
        """Which child functions of `node` to collapse, and which statements
        inside each of them to keep visible because they touch `wanted`
        identifiers. -> (stubs, excerpts)"""
        stubs = self.collapsible(node)
        excerpts: Dict[Node, List[Node]] = {}
        if not stubs or not wanted or depth >= 3:
            return stubs, excerpts
        owners: Dict[Node, List[Node]] = {}
        for ident in wanted:
            cur = ident.parent
            while cur is not None and cur is not node:
                if cur in stubs:
                    owners.setdefault(cur, []).append(ident)
                    break
                cur = cur.parent
        for fn, idents in owners.items():
            self._excerpts_for(fn, idents, stubs, excerpts, depth)
        return stubs, excerpts

    def _excerpts_for(self, fn, idents, stubs, excerpts, depth) -> None:
        """Keep the statements of collapsed function `fn` that touch `idents`."""
        stmts: List[Node] = []
        for ident in idents:
            st = _stmt_in(ident, fn)
            if st is not None and st not in stmts:
                stmts.append(st)
            if len(stmts) >= 3:
                break
        if not stmts:
            return
        excerpts[fn] = stmts
        for st in stmts:
            inner = {i for i in idents if self._within(i, st)}
            if st.kind in FUNCTION_KINDS and self.units[st].total >= self.collapse_nodes:
                stubs[st] = self.stub_text(st)
                if depth < 3:
                    self._excerpts_for(st, [i for i in inner], stubs, excerpts, depth + 1)
            s2, e2 = self.plan(st, inner, depth + 1)
            stubs.update(s2)
            excerpts.update(e2)

    @staticmethod
    def _within(node, ancestor) -> bool:
        cur = node
        while cur is not None:
            if cur is ancestor:
                return True
            cur = cur.parent
        return False

    # -- the code shown for an item --------------------------------------------
    def item_code(self, item: Item, budget: int, wanted: List[List[Node]], asked: Set[Unit]):
        root = item.root
        node = root.fn if root.fn is not None else self.session.tree.root
        markers = {m.fn: f'/* {m.id} */' for m in item.members if m.fn is not None}
        flat = {i for idents in wanted for i in idents}
        if len(item.members) > 1:
            stubs, excerpts = {}, {}
        else:
            stubs, excerpts = self.plan(node, flat)
        estimate = sum(m.own for m in item.members) * 5
        if estimate <= budget * 3:
            code = self.render(node, stubs, excerpts, markers)
            if len(code) <= budget:
                return code, False
        return self._slices(item, budget, wanted, asked), True

    def _owner_member(self, ident, members: Set[Unit]) -> Optional[Unit]:
        fn = _owner_fn(ident)
        while True:
            unit = self.units.get(fn)
            if unit in members:
                return unit
            if fn is None:
                return None
            fn = _owner_fn(fn)

    def _slices(self, item: Item, budget: int, wanted: List[List[Node]], asked: Set[Unit]) -> str:
        members = set(item.members)
        flat = {i for idents in wanted for i in idents}
        chosen: Dict[Node, Unit] = {}
        for idents in wanted:                        # every variable gets coverage first
            mine: List[Node] = []
            for ident in idents:
                unit = self._owner_member(ident, members)
                st = _stmt_in(ident, unit.fn) if unit is not None else None
                if st is None or st in mine:
                    continue
                mine.append(st)
                chosen.setdefault(st, unit)
                if len(mine) >= 3:
                    break
        keep = [st for st in chosen if not any(a in chosen for a in st.ancestors())]
        rendered: Dict[Node, str] = {}
        used = 0
        for st in keep:
            stubs, ex = self.plan(st, flat)
            if st.kind in FUNCTION_KINDS and self.units[st].total >= self.collapse_nodes:
                stubs[st] = self.stub_text(st)
                self._excerpts_for(st, [i for i in flat if self._within(i, st)], stubs, ex, 1)
            text = _clip(self.render(st, stubs, ex), 500)
            if used + len(text) > budget:
                break
            rendered[st] = text
            used += len(text) + 8
        root = item.root
        scan = root.fn if root.fn is not None else self.session.tree.root
        pos, need = {}, set(rendered)
        for i, n in enumerate(walk(scan)):
            if n in need:
                pos[n] = i
                need.discard(n)
                if not need:
                    break
        parts = []
        for unit in item.members:
            mine = sorted((st for st in rendered if chosen[st] is unit), key=lambda s: pos.get(s, 0))
            if not mine and unit not in asked:
                continue
            if unit.fn is not None:
                parts.append(self.render(unit.fn, {unit.fn: f'/* {unit.id} */'}))
            parts.extend(rendered[st] for st in mine)
        return '\n// ...\n'.join(parts)

    # -- context: where is this function used, what surrounds it ----------------
    def callsites(self, unit: Unit, members: Set[Unit], limit: int, cap: int, total: int) -> str:
        cands = []
        if unit.fn_binding is not None:
            seen = set()
            for ref in unit.fn_binding.reads:
                owner = self.units.get(_owner_fn(ref))
                if owner in members:
                    continue
                st = _stmt_in(ref, _owner_fn(ref))
                if st is not None and st not in seen:
                    seen.add(st)
                    cands.append((st, self.collapsible(st)))
                if len(cands) >= 12:
                    break
        elif unit.fn is not None and unit.parent is not None:
            parent = unit.fn.parent
            ctx = parent if parent is not None and parent.kind in (
                'Property', 'MethodDefinition', 'PropertyDefinition') else None
            if ctx is not None:
                cands.append((ctx, {**self.collapsible(ctx), unit.fn: '/* <this function> */'}))
            else:
                st = _stmt_in(unit.fn, unit.parent.fn)
                if st is not None:
                    cands.append((st, {**self.collapsible(st), unit.fn: '/* <this function> */'}))
        snippets = set()
        for node, stubs in cands:
            try:
                snippets.add(_clip(self.render(node, stubs), cap))
            except Exception:
                continue
        out, used = [], 0
        for snippet in sorted(snippets, key=len)[:limit]:
            if used + len(snippet) > total:
                break
            out.append(snippet)
            used += len(snippet)
        return '\n'.join(out)

    # -- skeleton for the overview ---------------------------------------------
    def words(self, unit: Unit) -> str:
        strings, props = [], Counter()
        for n in walk(unit.fn, kinds=('Literal', 'MemberExpression')):
            if n.kind == 'Literal':
                v = n.attr('value')
                if isinstance(v, str) and 3 <= len(v) <= 28 and v.strip() and v not in strings:
                    strings.append(v)
            elif not n.attr('computed', False):
                p = n.get('property')
                if p is not None and p.kind == 'Identifier' and len(p.attr('name')) > 2:
                    props[p.attr('name')] += 1
        for common in ('length', 'prototype', 'call', 'apply', 'push', 'slice'):
            props.pop(common, None)
        text = ''
        if strings:
            text += '; strings: ' + ', '.join(json.dumps(s) for s in strings[:4])
        if props:
            text += '; props: ' + ', '.join(p for p, _ in props.most_common(4))
        return text[:170].replace('*/', '')

    def skeleton(self, doc_order: List[Unit], budget: int) -> Tuple[str, int]:
        root = doc_order[0]
        expanded, added = {root}, []
        cost = root.own * 5
        heap = [(k.depth, k.own, k.index) for k in root.kids]
        heapq.heapify(heap)
        while heap:
            _d, own, idx = heapq.heappop(heap)
            if cost + own * 5 > budget:
                continue
            unit = doc_order[idx]
            expanded.add(unit)
            added.append(unit)
            cost += own * 5
            for k in unit.kids:
                heapq.heappush(heap, (k.depth, k.own, k.index))
        while True:
            stubs = {u.fn: f'/* {u.id}: {u.total} nodes{self.words(u)} */'
                     for u in doc_order if u not in expanded and u.parent in expanded}
            markers = {u.fn: f'/* {u.id} */' for u in expanded if u.fn is not None}
            code = self.render(self.session.tree.root, stubs, None, markers)
            if len(code) <= budget or not added:
                return code, len(expanded)
            excess = (len(code) - budget) * 1.3          # drop enough at once, not one by one
            while added and excess > 0:
                unit = added.pop()
                expanded.discard(unit)
                excess -= max(unit.own * 5, 1)


# ---------------------------------------------------------------------------
# Structural fingerprints (repeated code)
# ---------------------------------------------------------------------------

_SKIP_ATTRS = {'name', 'value', 'cooked', 'raw'}


def _fingerprint(unit: Unit, scopes, units) -> str:
    """Hash of a function's shape: node kinds, operators, structure, which
    variable is which (by order of appearance), strings and property names;
    but not identifier names or number values."""
    lo, hi = unit.index, unit.end
    local: Dict[object, int] = {}
    toks: List[str] = []
    stack = [(unit.fn, None, None)]
    while stack:
        n, role, parent = stack.pop()
        if n is None:
            toks.append(')')
            continue
        kind = n.kind
        if kind == 'Identifier':
            name = n.attr('name')
            if role == KEY and not parent.attr('computed', False):
                toks.append('K:' + name)
            elif role == LABEL:
                toks.append('L:' + name)
            else:
                b = scopes.binding_of(n)
                if b is None:
                    toks.append('T' if role == 'target' else 'G:' + name)
                else:
                    owner = units.get(b.scope.function)
                    if owner is not None and lo <= owner.index < hi:
                        toks.append(f'B{local.setdefault(b, len(local))}')
                    else:
                        toks.append('O')
            continue
        toks.append('(' + kind)
        for attr in attributes_of(kind):
            if attr in _SKIP_ATTRS:
                continue
            toks.append(f'{attr}={n.attr(attr)!r}')
        if kind == 'Literal':
            v = n.attr('value')
            if isinstance(v, bool):
                toks.append(f'b{v}')
            elif isinstance(v, float):
                toks.append('N')
            elif isinstance(v, str):
                toks.append('S:' + v[:60])
            elif isinstance(v, V.JSRegExp):
                toks.append(f'R:{v.pattern}/{v.flags}')
            else:
                toks.append('null')
        elif kind == 'TemplateElement':
            toks.append('Q:' + str(n.attr('raw'))[:60])
        stack.append((None, None, None))
        for _slot, child, crole in reversed(list(n.iter_children())):
            stack.append((child, crole, n))
    return hashlib.sha1('\x00'.join(toks).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

def _clock(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f'{h}:{m:02d}:{s:02d}' if h else f'{m}:{s:02d}'


class Progress:
    """One self-updating status line (\\r) with percent, request count, elapsed
    time and an ETA. Work is measured in 'asks' (names requested), which tracks
    generation time better than request count. The ETA is work done by requests
    that really hit the model divided by the wall-clock time during which at
    least one request was running, so it stays honest with several requests in
    flight and with cache hits mixed in. A ticker thread redraws every second.
    When stdout is not a terminal it prints a plain line every 30 s instead."""

    def __init__(self, enabled: bool, stream=None):
        self.enabled = enabled
        self.stream = stream or sys.stdout
        self.tty = bool(getattr(self.stream, 'isatty', lambda: False)())
        self.t0 = time.monotonic()
        self.total = 0.0
        self.done = 0.0
        self.fresh_work = 0.0
        self.busy_wall = 0.0
        self._busy_since: Optional[float] = None
        self.requests = 0
        self.planned = 0
        self.names = 0
        self.inflight: Dict[int, Tuple[str, float]] = {}
        self._next = 0
        self._last = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._width = 0

    def start(self):
        if self.enabled and self.tty and self._thread is None:
            self._thread = threading.Thread(target=self._tick, daemon=True)
            self._thread.start()

    def _tick(self):
        while not self._stop.wait(1.0):
            self.draw()

    def plan(self, work: float, requests: int = 1):
        self.total += work
        self.planned += requests

    def begin(self, label: str) -> int:
        with self._lock:
            token = self._next
            self._next += 1
            if not self.inflight:
                self._busy_since = time.monotonic()
            self.inflight[token] = (label, time.monotonic())
        self.draw(force=True)
        return token

    def end(self, token: int, work: float, cached: bool):
        with self._lock:
            self.inflight.pop(token, None)
            self.requests += 1
            self.done += work
            if not cached:
                self.fresh_work += work
            if not self.inflight and self._busy_since is not None:
                self.busy_wall += time.monotonic() - self._busy_since
                self._busy_since = None
        self.draw(force=True)

    def eta(self) -> Optional[float]:
        busy = self.busy_wall + (time.monotonic() - self._busy_since if self._busy_since else 0.0)
        if busy <= 0 or self.fresh_work <= 0:
            return None
        return max(self.total - self.done, 0.0) / (self.fresh_work / busy)

    def line(self) -> str:
        pct = min(100.0 * self.done / self.total if self.total else 100.0, 100.0)
        filled = int(pct / 100 * 20)
        eta = self.eta()
        flying = list(self.inflight.values())
        if len(flying) > 1:
            where = f'{len(flying)} in flight (oldest {int(time.monotonic() - min(s for _l, s in flying))}s)'
        elif flying:
            where = f'{flying[0][0]} ({int(time.monotonic() - flying[0][1])}s)'
        else:
            where = ''
        return (f'[{"#" * filled}{"-" * (20 - filled)}] {pct:3.0f}% | '
                f'{self.requests}/{self.planned} requests | {self.names} names | '
                f'{_clock(time.monotonic() - self.t0)} elapsed | '
                f'ETA {_clock(eta) if eta is not None else "--:--"} | {where}')

    def draw(self, force: bool = False):
        if not self.enabled:
            return
        with self._lock:
            now = time.monotonic()
            text = self.line()
            if self.tty:
                width = shutil.get_terminal_size((100, 20)).columns - 1
                text = text[:width]
                self.stream.write('\r' + text.ljust(self._width))
                self._width = len(text)
                self.stream.flush()
            elif force and now - self._last >= 30:
                self._last = now
                self.stream.write(text + '\n')
                self.stream.flush()

    def note(self, message: str):
        if not self.enabled:
            return
        with self._lock:
            if self.tty:
                self.stream.write('\r' + ' ' * self._width + '\r')
                self._width = 0
            self.stream.write(message + '\n')
            self.stream.flush()
        self.draw(force=True)

    def finish(self):
        self._stop.set()
        if self.enabled:
            with self._lock:
                if self.tty:
                    self.stream.write('\r' + ' ' * self._width + '\r')
                self.stream.flush()


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------

def _descriptor(binding, type_) -> str:
    bits = [binding.kind]
    if type_ and type_ != 'gen':
        bits.append(f'type: {type_}')
    bits.append(f'{len(list(binding.reads))} reads, {len(list(binding.writes))} writes')
    return f'- {binding.name}  ({", ".join(bits)})'


def _function_identity(binding):
    """The function node a binding *is*, if it is nothing else: a function
    declaration / named function expression, or a variable that is only ever
    assigned one function literal."""
    decl = binding.declaration
    if binding.kind == 'function':
        return decl if decl is not None and decl.kind in FUNCTION_KINDS else None
    if (binding.kind in ('var', 'let', 'const') and decl is not None
            and decl.kind == 'VariableDeclarator' and decl.get('id') is binding.target):
        init = decl.get('init')
        if (init is not None and init.kind in FUNCTION_KINDS
                and all(w is binding.target for w in binding.writes)):
            return init
    return None


def rename_vars_llm(session: Session,
                    llm: LocalLLM,
                    rename_globals: bool = False,
                    min_len_to_preserve: int = 0,
                    batch_size: int = 20,
                    max_prompt_chars: int = 12000,
                    cluster_chars: Optional[int] = None,
                    collapse_nodes: int = 50,
                    max_callsites: int = 3,
                    callsite_chars: int = 300,
                    neighbors: int = 0,
                    overview: bool = True,
                    overview_chars: Optional[int] = None,
                    match_clones: bool = True,
                    clone_min_nodes: int = 30,
                    clone_max_nodes: int = 12000,
                    repair: bool = True,
                    purpose_min_nodes: int = 8,
                    pack_chars: int = 0,
                    workers: int = 1,
                    trace_path: Optional[str] = None,
                    retries: int = 2,
                    unique_across_file: bool = False,
                    verbose: bool = True) -> None:
    """Rename variables and functions using a local LLM.

    :param max_prompt_chars: ceiling for system + user text of one request. JS
        tokenises at ~3 chars/token, so use about 2.2 * (context tokens - reply
        tokens): about 60000 for -c 32768.
    :param cluster_chars: a function whose whole subtree is estimated below this
        is sent as one request with all nested functions visible. Default 35%
        of max_prompt_chars. 0 disables clustering.
    :param collapse_nodes: nested functions with at least this many AST nodes
        are collapsed to header + purpose comment (variable uses inside them
        are still shown as excerpts).
    :param neighbors: how many already-named sibling functions to mention in a
        prompt. Off by default: a wrong name in one sibling then steers the rest.
    :param overview: for files that do not fit in one cluster, first ask for a
        description of the file and guesses for recognisable functions.
    :param overview_chars: skeleton budget (default: what fits max_prompt_chars).
    :param match_clones: copy names between structurally identical functions.
    :param repair: ask once more for names that would otherwise get a numeric
        suffix.
    :param pack_chars: >0 lets small independent items share one request.
    :param workers: requests in flight at once. Items are scheduled in
        dependency levels, so output is identical for any value; the server
        must offer that many slots (llama-server: --parallel N, and note that
        -c is divided between the slots).
    :param trace_path: write every request (prompt, reply, seconds) to this
        JSONL file, for working out why a name came out as it did.
    :param batch_size: max names asked per request.

    Afterwards: session.rename_map (new -> old, first wins), session.rename_log
    (every rename with origin 'llm', 'clone' or 'fallback'),
    session.function_notes (function name -> purpose), session.file_overview.
    """
    progress = Progress(verbose)
    if verbose:
        print("--- Applying Transformer: rename_vars_llm ---")
    scopes = session.scopes
    all_scopes = _scopes_top_down(scopes.root)
    exposed = _dynamic_exposed_scopes(all_scopes)
    if cluster_chars is None:
        cluster_chars = int(max_prompt_chars * 0.35)

    # -- 1. eligibility: identical rules to rename_vars -------------------------
    targets = {b.target for s in all_scopes for b in s.bindings.values() if b.target is not None}
    reserved = _collect_free_names(session, targets)
    eligible: Dict[object, List[object]] = {}
    for scope in all_scopes:
        for b in scope.bindings.values():
            skip = (b.implicit_global or b.name == 'arguments' or b.target is None
                    or (not rename_globals and scope is scopes.root)
                    or scope in exposed
                    or (min_len_to_preserve > 0 and len(b.name) >= min_len_to_preserve)
                    or _is_exported(b))
            if skip:
                reserved.add(b.name)
            else:
                eligible.setdefault(scope, []).append(b)
    eligible_set = {b for bs in eligible.values() for b in bs}

    # -- 2. types (before any edit), identifier nodes per binding ---------------
    inferer = TypeInferer(session)
    types = {b: inferer.binding(b) or 'gen' for b in eligible_set}
    nodes_of: Dict[object, Set[object]] = {}
    for ident in walk(session.tree.root, kinds=('Identifier',)):
        b = scopes.binding_of(ident)
        if b is not None:
            nodes_of.setdefault(b, set()).add(ident)

    def all_nodes(b) -> Set[object]:
        return set(b.reads) | set(b.writes) | nodes_of.get(b, set()) | {b.target}

    # -- 3. units: sizes, order, dependencies ------------------------------------
    root_unit = Unit(None)
    units: Dict[object, Unit] = {None: root_unit}
    doc_order = [root_unit]
    stack = [(session.tree.root, root_unit)]
    while stack:
        n, owner = stack.pop()
        if n.kind in FUNCTION_KINDS:
            unit = Unit(n)
            unit.parent = owner
            unit.depth = owner.depth + 1
            owner.kids.append(unit)
            units[n] = unit
            doc_order.append(unit)
            owner = unit
        owner.own += 1
        stack.extend((c, owner) for _s, c, _r in reversed(list(n.iter_children())))
    for i, unit in enumerate(doc_order):
        unit.index, unit.id = i, f'F{i}'
    for unit in reversed(doc_order):
        unit.total += unit.own
        if unit.parent is not None:
            unit.parent.total += unit.total
            unit.parent.size += unit.size
            unit.parent.deps[unit] = None

    for scope in all_scopes:
        units[scope.function].scopes.append(scope)
        for b in scope.bindings.values():
            fn = _function_identity(b)
            if fn is not None and fn in units and units[fn].fn_binding is None:
                units[fn].fn_binding = b
                if b in eligible_set:
                    units[fn].name_binding = (scope, b)
    name_bindings = {u.name_binding[1] for u in doc_order if u.name_binding}
    for unit in doc_order:
        unit.variables = sorted(
            ((s, b) for s in unit.scopes for b in eligible.get(s, ()) if b not in name_bindings),
            key=lambda sb: sb[1].kind != 'param')
        for sb in unit.variables:
            unit.by_name.setdefault(sb[1].name, []).append(sb)
        if unit.fn_binding is not None:
            for ref in unit.fn_binding.reads:
                caller = units.get(_owner_fn(ref))
                if caller is not None and caller is not unit:
                    caller.deps[unit] = None

    # -- 4. repeated code: name one copy, copy the names ------------------------
    aliases: List[Tuple[Unit, Unit]] = []        # (representative, clone)

    def compatible(r: Unit, c: Unit) -> bool:
        return (len(r.variables) == len(c.variables)
                and all(x[1].kind == y[1].kind for x, y in zip(r.variables, c.variables))
                and (r.name_binding is None) == (c.name_binding is None))

    if match_clones:
        groups: Dict[str, List[Unit]] = {}
        for unit in doc_order[1:]:
            if clone_min_nodes <= unit.total <= clone_max_nodes:
                groups.setdefault(_fingerprint(unit, scopes, units), []).append(unit)
        for group in sorted((g for g in groups.values() if len(g) > 1),
                            key=lambda g: -g[0].total):
            live = [u for u in group if not u.aliased]
            if len(live) < 2:
                continue
            rep = live[0]
            for clone in live[1:]:
                if rep.index <= clone.index < rep.end or clone.aliased:
                    continue                      # nested inside the representative
                pairs = list(zip(doc_order[rep.index:rep.end], doc_order[clone.index:clone.end]))
                if rep.size != clone.size or not all(compatible(r, c) for r, c in pairs):
                    continue
                if any(c.aliased for _r, c in pairs):
                    continue
                for _r, c in pairs:
                    c.aliased = True
                aliases.append((rep, clone))

    # -- 5. items: clusters and singletons --------------------------------------
    fit_nodes = cluster_chars // 5
    roots: List[Unit] = []
    cluster_of: Dict[Unit, Unit] = {}
    for unit in doc_order:
        p = unit.parent
        if cluster_chars > 0 and p is not None and p.total <= fit_nodes:
            cluster_of[unit] = cluster_of[p]
        else:
            cluster_of[unit] = unit
            roots.append(unit)
    items: Dict[Unit, Item] = {}
    for r in roots:
        fits = cluster_chars > 0 and r.total <= fit_nodes
        items[r] = Item(r, doc_order[r.index:r.end] if fits else [r])
        for m in items[r].members:
            m.item = items[r]
    for item in items.values():
        for m in item.members:
            for d in m.deps:
                if d.item is not item:
                    item.deps[d.item] = None
            if m.aliased:
                continue
            if m.fn is not None:
                if m.name_binding is not None:
                    item.asks.append((m, '@name'))
                if m.name_binding is not None or m.by_name or m.total >= purpose_min_nodes:
                    item.asks.append((m, '@purpose'))
            item.asks.extend((m, name) for name in m.by_name)
    for rep, clone in aliases:
        if clone.item is not rep.item:
            clone.item.deps[rep.item] = None
    order = _dependency_order([items[r] for r in roots], lambda it: it.deps)

    # Items are scheduled in dependency levels: everything in a level only depends on
    # earlier levels, so a level can run concurrently and the result is the same for
    # any number of workers. Chunks of one item occupy consecutive levels.
    jobs: List[Job] = []
    position = {item: i for i, item in enumerate(order)}
    done_level: Dict[Item, int] = {}
    for item in order:
        base = 0
        for d in item.deps:
            if position[d] < position[item] and d in done_level:
                base = max(base, done_level[d] + 1)
        chunks = [item.asks[i:i + batch_size] for i in range(0, len(item.asks), batch_size)]
        for i, ch in enumerate(chunks):
            job = Job(item, ch, last=(i == len(chunks) - 1))
            job.level = base + i
            jobs.append(job)
        done_level[item] = base + len(chunks) - 1 if chunks else base - 1

    # -- shared machinery ----------------------------------------------------------
    renderer = Renderer(session, units, collapse_nodes)
    table = NameTable(reserved, all_scopes, lambda n: scopes.scope_of(n), all_nodes, unique_across_file)
    log: List[dict] = []
    entry_of: Dict[object, dict] = {}
    state = {'file': None, 'repairs': 0}
    cache0 = getattr(llm, 'cache_hits', 0)
    timings: List[Tuple[float, str, int, bool]] = []
    trace_lock = threading.Lock()
    info = getattr(llm, 'complete_info', None)

    def call(system, user, attempt, max_tokens):
        validate = lambda text: bool(parse_json_object(text))
        if info is not None:
            return info(system, user, attempt, max_tokens, validate)
        return llm.complete(system, user, attempt, max_tokens), False, 0.0

    def fetch(system: str, user: str, label: str, work: float, max_tokens=None) -> dict:
        """One logical request (with retries). Touches no tree state, so it can
        run on a worker thread."""
        token = progress.begin(label)
        data, text, seconds, cached = {}, '', 0.0, True
        try:
            for attempt in range(retries + 1):
                text, was_cached, took = call(system, user, attempt, max_tokens)
                seconds += took
                cached = cached and was_cached
                data = parse_json_object(text)
                if data:
                    break
        finally:
            progress.end(token, work, cached)
        with trace_lock:
            timings.append((seconds, label, len(system) + len(user), cached))
            if trace_path:
                with open(trace_path, 'a', encoding='utf-8') as handle:
                    handle.write(json.dumps({'label': label, 'seconds': round(seconds, 2),
                                             'cached': cached, 'prompt': user, 'reply': text}) + '\n')
        return data

    def run_all(tasks):
        if workers <= 1 or len(tasks) <= 1:
            return [task() for task in tasks]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(task) for task in tasks]
            return [f.result() for f in futures]

    def commit(scope, b, wanted, forced_type, origin='llm') -> str:
        if wanted:
            new, how = table.claim(scope, b, wanted), origin
        else:
            stem = 'arg' if b.kind == 'param' else TYPE_STEMS[forced_type or types.get(b, 'gen')]
            new, how = table.claim(scope, b, stem, numbered=True), 'fallback'
        for ident in all_nodes(b):
            if ident.live:
                ident.attrs['name'] = new
        entry = entry_of.get(b)
        if entry is None:
            entry = entry_of[b] = {'old': b.name, 'kind': b.kind}
            log.append(entry)
        entry['new'], entry['origin'] = new, how
        progress.names = len(log)
        return new

    def proposal(raw, b, second_pass: bool = False) -> Optional[str]:
        """A usable proposed name, or None. Handing back the minified name is
        'no answer' the first time; on the follow-up a plain short name such as
        `i` is accepted but anything with a digit (u8, h68) is not."""
        name = sanitize_identifier(raw)
        if name and is_echo(name, b.name) and (not second_pass or any(c.isdigit() for c in b.name)):
            return None
        return name

    def unit_reply(data: dict, unit: Unit, single_request: bool) -> Optional[dict]:
        """This unit's part of a reply. Accepts the flat form {"a": "x"} when the
        request concerned exactly one function and the model skipped the nesting."""
        sub = data.get(unit.id)
        if isinstance(sub, dict):
            return sub
        if single_request and data and not any(isinstance(v, dict) for v in data.values()):
            return data
        return None

    def apply(job: Job, data: dict, single_request: bool):
        """Write the model's names into the tree. -> follow-ups needed:
        (unit, key, scope, binding, rejected_proposal_or_None, 'clash'|'missing')"""
        pending, got = [], False

        def settle(unit, key, scope, b, wanted, forced):
            new = commit(scope, b, wanted, forced)
            if wanted is None:
                pending.append((unit, key, scope, b, None, 'missing'))
            elif new != wanted:
                pending.append((unit, key, scope, b, wanted, 'clash'))

        for unit, key in job.asks:
            sub = unit_reply(data, unit, single_request)
            got = got or bool(sub)
            val = sub.get(key) if sub else None
            if key == '@purpose':
                purpose = sanitize_text(val)
                if purpose:
                    unit.purpose = purpose
            elif key == '@name':
                scope, b = unit.name_binding
                settle(unit, key, scope, b, proposal(val, b), 'fn')
            else:
                for scope, b in unit.by_name[key]:
                    settle(unit, key, scope, b, proposal(val, b), None)
        return pending if got else []          # a wholly unusable reply was already retried

    def repair_request(job: Job, pending) -> Tuple[str, str]:
        by_unit: Dict[Unit, list] = {}
        for p in pending:
            by_unit.setdefault(p[0], []).append(p)
        lines, taken = [], []
        for unit, group in by_unit.items():
            lines.append(f'[{unit.id}]')
            for _u, key, scope, b, wanted, why in group:
                if why == 'clash':
                    lines.append(f'- {key}: `{wanted}` is already used by another variable here; '
                                 f'say what makes this one different')
                else:
                    lines.append(f'- {key}: no usable name was given (missing, or the old name repeated); '
                                 f'give a descriptive one')
                taken.extend(table.nearby(scope, 25))
        user = (prefix() + job.view +
                'Some names could not be used. Give each item below a new, specific name:\n' +
                '\n'.join(lines) +
                '\nNames already in use nearby (do not reuse): ' + ', '.join(sorted(set(taken))[:50]) +
                '\nReply with one JSON object of the same shape containing only these items.')
        return user, f'follow-up for {len(pending)} names'

    def apply_repair(pending, data: dict) -> None:
        single = len({p[0] for p in pending}) == 1
        flat = bool(data) and not any(isinstance(v, dict) for v in data.values())
        keys = Counter(p[1] for p in pending)
        for unit, key, scope, b, wanted, why in pending:
            sub = unit_reply(data, unit, single)
            if sub is not None:
                raw = sub.get(key)
            else:                                   # flat reply covering several functions
                raw = data.get(key) if flat and keys[key] == 1 else None
            name = proposal(raw, b, second_pass=True)
            if name and name != wanted:
                commit(scope, b, name, 'fn' if key == '@name' else None)
                state['repairs'] += 1

    def apply_aliases(members: List[Unit]) -> None:
        reps = set(members)
        for rep, clone in aliases:
            if rep not in reps or clone.copied:
                continue
            clone.copied = True
            for r, c in zip(doc_order[rep.index:rep.end], doc_order[clone.index:clone.end]):
                c.purpose = r.purpose
                if r.name_binding is not None:
                    commit(c.name_binding[0], c.name_binding[1],
                           table.name_of(r.name_binding[1]), 'fn', origin='clone')
                for (_sr, br), (sc, bc) in zip(r.variables, c.variables):
                    commit(sc, bc, table.name_of(br), None, origin='clone')

    # -- context text ----------------------------------------------------------------
    def current_name(unit: Unit) -> Optional[str]:
        return unit.fn_binding.target.attrs['name'] if unit.fn_binding is not None else None

    def slot_label(unit: Unit) -> str:
        """How to refer to a function: its name, else where it is stored (z.d,
        a property key), else '(anonymous)'."""
        name = current_name(unit)
        if name:
            return name
        parent = unit.fn.parent if unit.fn is not None else None
        try:
            if parent is not None and parent.kind == 'AssignmentExpression' \
                    and parent.get('right') is unit.fn:
                return _clip(renderer.render(parent.get('left')), 40)
            if parent is not None and parent.kind in ('Property', 'MethodDefinition',
                                                      'PropertyDefinition'):
                key = renderer.render(parent.get('key'))
                return _clip(f'.{key}' if parent.kind != 'Property' else key, 40)
        except Exception:
            pass
        return '(anonymous)'

    def crumbs(unit: Unit) -> str:
        chain, cur = [], unit.parent
        while cur is not None and cur.fn is not None and len(chain) < 3:
            note = cur.hint_purpose
            if note:
                chain.append(f'[{cur.id}] {cur.hint_name + ": " if cur.hint_name else ""}{note}')
            cur = cur.parent
        return ' > '.join(reversed(chain))

    def nearby_text(unit: Unit) -> str:
        if neighbors <= 0 or unit.parent is None:
            return ''
        sibs = sorted((k for k in unit.parent.kids if k is not unit and k.purpose),
                      key=lambda k: abs(k.index - unit.index))[:neighbors]
        return '\n'.join(f'- {slot_label(k)}: {k.purpose}'
                         for k in sorted(sibs, key=lambda k: k.index))

    def prefix() -> str:
        return f'File: {state["file"]}\n\n' if state['file'] else ''

    # -- overview --------------------------------------------------------------------
    file_overview = None
    big = doc_order[0].total * 5 > cluster_chars if cluster_chars > 0 else len(doc_order) > 1
    if overview and big and jobs:
        budget = overview_chars or max(2000, max_prompt_chars - len(OVERVIEW_PROMPT) - 700)
        progress.plan(25)
        progress.start()
        skeleton, shown = renderer.skeleton(doc_order, budget)
        user = ('Structure of the file:\n```js\n' + skeleton + '\n```\n\n'
                'Reply with the JSON object described.')
        data = fetch(OVERVIEW_PROMPT, user, f'overview ({shown} functions in view)', 25, 2048)
        file_overview = sanitize_text(data.get('@file'), 400)
        state['file'] = file_overview
        by_id = {u.id: u for u in doc_order}
        for key, val in data.items():
            if key in by_id and isinstance(val, dict):
                by_id[key].hint_name = sanitize_identifier(val.get('@name'))
                by_id[key].hint_purpose = sanitize_text(val.get('@purpose'))
    progress.plan(sum(len(j.asks) + 3 for j in jobs), len(jobs))
    progress.start()

    # -- build one job's prompt text -----------------------------------------------------
    def build(job: Job) -> str:
        item, root = job.item, job.item.root
        members = set(item.members)
        groups: List[Tuple[Unit, List[str]]] = []
        for unit, key in job.asks:
            if not groups or groups[-1][0] is not unit:
                groups.append((unit, []))
            groups[-1][1].append(key)
        wanted, lines = [], []
        for unit, keys in groups:
            head = f'[{unit.id}]'
            if unit.hint_purpose:
                guess = f'"{unit.hint_name}" - ' if unit.hint_name else ''
                head += f'  (overview guess: {guess}{unit.hint_purpose})'
            lines.append(head)
            for key in keys:
                if key == '@name':
                    lines.append('- @name  (this function itself)')
                elif key == '@purpose':
                    lines.append('- @purpose  (one sentence on what this function does)')
                else:
                    b = unit.by_name[key][0][1]
                    lines.append(_descriptor(b, types.get(b, 'gen')))
                    wanted.append(sorted(all_nodes(b), key=lambda n: n is not b.target)[:12])
        ask_text = '\n'.join(lines)
        calls = renderer.callsites(root, members, max_callsites, callsite_chars,
                                   callsite_chars * max_callsites)
        extra = ''
        trail = crumbs(root)
        if trail:
            extra += f'Located in: {trail}\n'
        near = nearby_text(root)
        if near:
            extra += f'Nearby functions already understood:\n{near}\n'
        room = (max_prompt_chars - len(SYSTEM_PROMPT) - len(prefix()) - 400
                - len(ask_text) - len(calls) - len(extra))
        code, _sliced = renderer.item_code(item, max(800, room), wanted, {u for u, _ in groups})
        label = 'Top-level code' if root.fn is None else ('Code' if len(item.members) > 1 else 'Function')
        view = f'{extra}{label}:\n```js\n{code}\n```\n'
        if calls:
            view += f'Where it is used:\n```js\n{calls}\n```\n'
        job.view = view
        return view + f'Name these:\n{ask_text}\n'

    def make_packs(level_jobs: List[Job]) -> List[List[Job]]:
        packs, i = [], 0
        while i < len(level_jobs):
            pack, used = [], len(SYSTEM_PROMPT) + len(prefix())
            while i < len(level_jobs):
                job = level_jobs[i]
                job.text = build(job)
                if pack and (len(job.text) > pack_chars or used + len(job.text) + 60 > max_prompt_chars):
                    break
                pack.append(job)
                i += 1
                used += len(job.text) + 20
                if not pack_chars or len(job.text) > pack_chars:
                    break
            packs.append(pack)
        return packs

    # -- run, one dependency level at a time -------------------------------------------
    by_level: Dict[int, List[Job]] = {}
    for job in jobs:
        by_level.setdefault(job.level, []).append(job)
    requests = packed = 0
    for level in sorted(by_level):
        packs = make_packs(by_level[level])       # every prompt is built before any name is applied
        tasks, metas = [], []
        for pack in packs:
            ids = sorted({u.id for j in pack for u, _ in j.asks}, key=lambda s: int(s[1:]))
            user = prefix() + '\n'.join(f'### {j.item.root.id}\n{j.text}' for j in pack)
            user += '\nReply with one JSON object. Keys: ' + ', '.join(f'"{i}"' for i in ids) + '.'
            work = sum(len(j.asks) + 3 for j in pack)
            label = ', '.join(j.item.root.id for j in pack[:3]) + (f' +{len(pack) - 3}' if len(pack) > 3 else '')
            tasks.append(lambda user=user, label=label, work=work: fetch(SYSTEM_PROMPT, user, label, work))
            metas.append((pack, len(ids) == 1))
        results = run_all(tasks)
        requests += len(packs)
        packed += sum(len(p) - 1 for p in packs)

        follow_ups = []
        for (pack, single), data in zip(metas, results):
            for j in pack:
                pending = apply(j, data, single)
                if repair and pending:
                    follow_ups.append((j, pending))
        if follow_ups:
            tasks = []
            for j, pending in follow_ups:
                user, label = repair_request(j, pending)
                progress.plan(len(pending) + 3, 1)
                tasks.append(lambda user=user, label=label, n=len(pending):
                             fetch(SYSTEM_PROMPT, user, label, n + 3))
            requests += len(tasks)
            for (j, pending), data in zip(follow_ups, run_all(tasks)):
                apply_repair(pending, data)
        for pack in packs:
            for j in pack:
                if j.last:
                    j.item.done = True
                    apply_aliases(j.item.members)
    apply_aliases(doc_order)                       # clones whose representative had no job

    # -- finish --------------------------------------------------------------------------
    rename_map: Dict[str, str] = {}
    for e in log:
        rename_map.setdefault(e['new'], e['old'])
    notes = {}
    for u in doc_order:
        if u.purpose and current_name(u):
            notes[current_name(u)] = u.purpose
    session.rename_map, session.rename_log = rename_map, log
    session.function_notes, session.file_overview = notes, file_overview
    if log:
        _fix_shorthand_properties(session.tree.root)
        session.tree._touch()
        scopes.mark_dirty()
    progress.finish()
    origin = Counter(e['origin'] for e in log)
    if verbose:
        print(f" |-> Renamed {len(log)} variables in {_clock(time.monotonic() - progress.t0)}: "
              f"{requests} requests ({getattr(llm, 'cache_hits', 0) - cache0} cached, {packed} packed), "
              f"{len(aliases)} cloned functions, {state['repairs']} repaired clashes. "
              f"Origin: llm={origin['llm']} clone={origin['clone']} fallback={origin['fallback']}")
        slow = sorted((t for t in timings if not t[3]), reverse=True)[:3]
        if slow:
            print(" |-> Model time " + _clock(progress.busy_wall) + " of " +
                  _clock(time.monotonic() - progress.t0) + "; slowest requests: " +
                  ', '.join(f'{lab} {sec:.0f}s ({chars // 1000}k chars)' for sec, lab, chars, _c in slow))
        if file_overview:
            print(f" |-> Overview: {file_overview}")
