import re
import string
from collections import Counter
from math import gcd
from typing import Dict, List, Optional, Set

from jsir import Session, walk, values as V
from jsir.schema import FUNCTION_KINDS, slots_of, VALUE, KEY, LABEL

# ---------------------------------------------------------------------------
# Naming scheme
#
#   function            fn_<key>              fn_S      fn_Zq
#   parameter           ag_<key><n>           ag_S0 ag_S1 ag_S2      (n = position)
#   other local         <tag>_<key><n>        st_S0  nm_S0  nm_S1  ar_S0
#   top-level (no key)  <tag>_<n>             st_0
#
# * <tag>  two letters naming the inferred type (TAGS below). Parameters always
#          use PARAM_TAG so they are easy to follow locally; set it to None to
#          type them like any other variable.
# * <key>  a letters-only id, one per function, handed out in a scrambled order
#          so neighbouring functions look nothing alike. Keys start at one
#          letter and grow to two (then three) only when the shorter pool is
#          used up. By default the key set is prefix-free (no key is the start
#          of another), so searching `_S` finds exactly one function's names.
# * <n>    counts per (function, tag), starting at 0 in every function.
#
# Keys are unique per function and the counter is unique per (function, tag),
# so every generated name is unique across the file. That also makes capture
# impossible without any scope-chain bookkeeping.
# ---------------------------------------------------------------------------

TAGS = {
    'num': 'num', 'str': 'str', 'arr': 'arr', 'obj': 'obj', 'bool': 'bool',
    'fn': 'fn', 'cls': 'cls', 're': 'rx', 'err': 'err', 'gen': 'gen',
}
PARAM_TAG = 'arg'
FUNCTION_TAG = TAGS['fn']
LOOKALIKE_LETTERS = 'lIO'


def _stride(size: int) -> int:
    """A step near size/phi that is coprime with size: i -> i*step % size is a
    permutation that keeps successive values far apart."""
    step = max(1, round(size * 0.6180339887498949))
    while gcd(step, size) != 1:
        step += 1
    return step


def _plan_keys(n: int, base: int, prefix_free: bool):
    """How many keys of each length: [(length, first_code, count), ...].

    prefix_free=True  canonical prefix code: as many one-letter keys as will
                      still leave room for the rest within the shortest
                      possible maximum length. No key starts another key.
    prefix_free=False every one-letter key, then every two-letter key, ...
                      (shortest names, but `_S` also matches `_Sa`)."""
    plan = []
    if n <= 0:
        return plan
    if not prefix_free:
        length, remaining = 1, n
        while remaining > 0:
            count = min(base ** length, remaining)
            plan.append((length, 0, count))
            remaining -= count
            length += 1
        return plan

    longest = 1
    while base ** longest < n:
        longest += 1
    first, remaining = 0, n
    for length in range(1, longest + 1):
        avail = base ** length - first
        spare = base ** (longest - length)        # keys one prefix can still hold
        if length == longest or remaining <= avail:
            count = remaining
        else:
            count = max(0, min(avail, (avail * spare - remaining) // (spare - 1)))
        if count:
            plan.append((length, first, count))
        remaining -= count
        if remaining <= 0:
            break
        first = (first + count) * base
    return plan


class KeySeq:
    """Never-repeating keys: the one-letter pool first, then two letters, and
    so on, each pool in scrambled order."""

    def __init__(self, plan, alphabet: str, taboo: Set[str]):
        self.plan = plan
        self.alphabet = alphabet
        self.taboo = taboo
        self.level = 0
        self.j = 0

    def next(self) -> str:
        base = len(self.alphabet)
        while self.level < len(self.plan):
            length, first, count = self.plan[self.level]
            if self.j >= count:
                self.level += 1
                self.j = 0
                continue
            idx = ((self.j + count // 7) * _stride(count)) % count
            self.j += 1
            code = first + idx
            letters = []
            for _ in range(length):
                code, d = divmod(code, base)
                letters.append(self.alphabet[d])
            key = ''.join(reversed(letters))
            if key not in self.taboo:
                return key
        raise RuntimeError('key space exhausted')


class Namer:
    def __init__(self, reserved: Set[str], n_functions: int,
                 avoid_lookalikes: bool = True, prefix_free: bool = True):
        alphabet = ''.join(c for c in string.ascii_letters
                           if not (avoid_lookalikes and c in LOOKALIKE_LETTERS))
        # Keys that would produce a name already present in the source are never used.
        tags = '|'.join(sorted(set(TAGS.values()) | {PARAM_TAG} - {None}))
        shape = re.compile(rf'^(?:{tags})_([A-Za-z]+)\d*$')
        taboo = {m.group(1) for name in reserved if (m := shape.match(name))}

        self.reserved = reserved
        self.seq = KeySeq(_plan_keys(n_functions + len(taboo), len(alphabet), prefix_free),
                          alphabet, taboo)
        self.keys: Dict[object, str] = {}
        self.origin: Dict[object, Optional[str]] = {}
        self.claimed: Set[object] = set()
        self.counters: Dict[object, int] = {}

    def key_of(self, function) -> str:
        key = self.keys.get(function)
        if key is None:
            key = self.keys[function] = self.seq.next()
        return key

    def claim_function(self, function, old_name: str) -> bool:
        """A function node gets the name fn_<key> once; later bindings to the
        same node fall back to numbered locals."""
        if function in self.claimed:
            return False
        self.claimed.add(function)
        self.origin[function] = old_name
        return True

    def function_name(self, function) -> str:
        return f'{FUNCTION_TAG}_{self.key_of(function)}'

    def local_name(self, owner, tag: str) -> str:
        """owner is the enclosing function node, or None at top level."""
        key = self.key_of(owner) if owner is not None else ''
        n = self.counters.get((owner, tag), 0)
        while True:
            name = f'{tag}{n}_{key}'
            n += 1
            if name not in self.reserved:
                break
        self.counters[(owner, tag)] = n
        return name


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
# The pass
# ---------------------------------------------------------------------------

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


def rename_vars(session: Session,
                rename_globals: bool = False,
                min_len_to_preserve: int = 0,
                avoid_lookalikes: bool = True,
                prefix_free: bool = True) -> None:
    """Renames variables to <tag>_<function key><n> (see the naming scheme).

    :param session: The active jsir Session.
    :param rename_globals: Also rename bindings in the Program root scope
        (they get key-less names such as st_0).
    :param min_len_to_preserve: If > 0, keep names at least this long
        (already-deobfuscated names).
    :param avoid_lookalikes: Leave l, I and O out of function keys.
    :param prefix_free: True = no key is the start of another key, so a plain
        substring search for `_S` finds only that function (costs a few
        one-letter keys). False = use every one-letter key before moving to
        two letters.

    Afterwards `session.rename_map` maps new name -> old name and
    `session.function_keys` maps key -> the function's original name (or None).
    """
    print("--- Applying Transformer: rename_vars ---")

    scopes = session.scopes
    all_scopes = _scopes_top_down(scopes.root)
    exposed = _dynamic_exposed_scopes(all_scopes)

    # -- 1. decide what is renamed; everything else is pinned ---------------
    targets: Set[object] = set()
    for scope in all_scopes:
        for b in scope.bindings.values():
            if b.target is not None:
                targets.add(b.target)

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
                reserved.add(b.name)          # untouched, so nothing may collide with it
            else:
                eligible.setdefault(scope, []).append(b)

    # -- 2. infer every type before touching the tree -----------------------
    inferer = TypeInferer(session)
    types: Dict[object, str] = {}
    identity: Dict[object, object] = {}
    needs_key: Set[object] = set()
    for scope, bindings in eligible.items():
        if scope.function is not None:
            needs_key.add(scope.function)
        for b in bindings:
            types[b] = inferer.binding(b) or 'gen'
            fn = _function_identity(b)
            if fn is not None:
                identity[b] = fn
                needs_key.add(fn)

    # -- 3. every identifier node that belongs to each binding ----------------
    nodes_of: Dict[object, Set[object]] = {}
    for ident in walk(session.tree.root, kinds=('Identifier',)):
        b = scopes.binding_of(ident)
        if b is not None:
            nodes_of.setdefault(b, set()).add(ident)

    # -- 4. allocate parents-first and rewrite in place ----------------------
    namer = Namer(reserved, len(needs_key), avoid_lookalikes, prefix_free)
    rename_map: Dict[str, str] = {}
    counts: Counter = Counter()

    for scope in all_scopes:
        # parameters first, so ag_<key><n> follows their order in the signature
        ordered = sorted(eligible.get(scope, ()), key=lambda b: b.kind != 'param')
        for b in ordered:
            fn = identity.get(b)
            if fn is not None and namer.claim_function(fn, b.name):
                new_name = namer.function_name(fn)
                tag = FUNCTION_TAG
            else:
                tag = PARAM_TAG if (b.kind == 'param' and PARAM_TAG) else TAGS[types[b]]
                new_name = namer.local_name(scope.function, tag)

            nodes = set(b.reads) | set(b.writes) | nodes_of.get(b, set())
            nodes.add(b.target)
            for ident in nodes:
                if ident.live:
                    ident.attrs['name'] = new_name
            rename_map[new_name] = b.name
            counts[tag] += 1

    renamed = len(rename_map)
    session.rename_map = rename_map
    session.function_keys = {key: namer.origin.get(fn) for fn, key in namer.keys.items()}

    if renamed > 0:
        _fix_shorthand_properties(session.tree.root)
        session.tree._touch()
        scopes.mark_dirty()

    lengths = Counter(len(k) for k in namer.keys.values())
    print(f" |-> Renamed {renamed} variables across {len(namer.keys)} functions "
          f"(keys by length: {dict(sorted(lengths.items()))}). By tag: {dict(counts)}")
