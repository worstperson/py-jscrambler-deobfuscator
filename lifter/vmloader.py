"""
vmloader -- decode VM payloads by interpreting the loader found in the sample.

Nothing about a build's header format is hard-coded. The factory function that
wraps the interpreter starts with a loader: it copies the payload string into a
byte array, derives a key, reads header fields, a constant pool and an
exception table, and records where the code starts. This module

    1. finds the factory, its call site(s) and the payload string(s)
    2. extracts the loader statements (everything before the interpreter)
    3. runs them over the payload in a small whitelisted JS-subset interpreter
       (pure Python: no I/O, no eval, step-budgeted; anything unsupported is a
       hard error rather than a guess)
    4. reads the results back out by ROLE, not by name:
         constants   the array the CONSTS handlers index          (vmsummary roles)
         exceptions  the array indexed [i-3..i] by the catch-handler's lookup
         key         the 4th argument at the interpreter's call site
         code base   the offset added to jump targets by the jump helper
         flags       the variable the factory guards its result with  (== N)
         name/arity  the Object.defineProperty(.., 'name'|'length', ..) epilogue

so a rebuilt loader (different section order, tags, key seed, helpers) decodes
without changes. Child VMs are string constants decoded by the same loader.

Usage:  python vmloader.py SCRIPT.js [--consts N]
"""

import math
import struct
import sys
from urllib.parse import unquote

from jsir import Session, walk, values as V
from jsir.schema import FUNCTION_KINDS
from lifter.vmsummary import Interpreter, analyse, _name, CONSTS


class LoaderError(Exception):
    pass


# ---------------------------------------------------------------------------
# A deliberately tiny JS-subset machine
# ---------------------------------------------------------------------------

UNDEF, NULL = V.UNDEF, V.NULL


class Native:
    def __init__(self, fn):
        self.fn = fn


class NativeCtor:
    def __init__(self, construct):
        self.construct = construct


class Buffer:
    def __init__(self, size):
        self.data = bytearray(size)


class TypedArray:
    def __init__(self, kind, buf, trace=None):
        self.kind, self.buf, self.trace = kind, buf, trace

    @property
    def length(self):
        return len(self.buf.data) // (8 if self.kind == 'f64' else 1)

    def get(self, i):
        if not 0 <= i < self.length:
            return UNDEF
        if self.trace is not None:
            self.trace.append(i)
        if self.kind == 'f64':
            return struct.unpack_from('<d', self.buf.data, 8 * i)[0]
        return float(self.buf.data[i])

    def set(self, i, value):
        if self.kind != 'u8' or not 0 <= i < self.length:
            raise LoaderError('unsupported typed-array store')
        bits = V.to_uint32(value)
        if bits is V.UNKNOWN:
            raise LoaderError('non-numeric store into a byte array')
        self.buf.data[i] = bits & 0xFF


class JSThrow(Exception):
    def __init__(self, value):
        self.value = value


class JSError:
    pass


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


class _Return(Exception):
    def __init__(self, value):
        self.value = value


class Closure:
    def __init__(self, node, env):
        self.node, self.env = node, env


class Env:
    __slots__ = ('vars', 'parent')

    def __init__(self, parent=None):
        self.vars, self.parent = {}, parent

    def find(self, name):
        env = self
        while env is not None:
            if name in env.vars:
                return env
            env = env.parent
        return None


def _key(k):
    if isinstance(k, float) and k == k and k not in (math.inf, -math.inf) and k == int(k):
        return int(k)
    return k


class Machine:
    def __init__(self, payload_len, budget=5_000_000):
        self.budget = budget
        self.payload_len = payload_len
        self.trace = []                      # byte offsets read from the payload array
        self._traced = False
        g = Env()
        self.global_env = g
        u8 = lambda arg=0.0: self._new_u8(arg)
        namespaces = {
            'Uint8Array': NativeCtor(u8),
            'ArrayBuffer': NativeCtor(lambda n: Buffer(int(n))),
            'Float64Array': NativeCtor(lambda b: TypedArray('f64', b)),
            'Error': NativeCtor(lambda *a: JSError()),
            'String': {'fromCharCode': Native(lambda *c: ''.join(chr(int(x) & 0xFFFF) for x in c))},
        }
        self.namespaces = namespaces
        for name, value in namespaces.items():
            g.vars[name] = value
        g.vars['_global'] = self._global_stub()

    def _global_stub(self):
        stub = {}
        for name, value in self.namespaces.items():
            stub[name] = Native(lambda v=value: v)       # `_global.Uint8Array()` returns the constructor
        return _GlobalStub(stub)

    def _new_u8(self, arg):
        if isinstance(arg, Buffer):
            return TypedArray('u8', arg)
        size = int(arg)
        arr = TypedArray('u8', Buffer(size))
        if size == self.payload_len and not self._traced:
            arr.trace, self._traced = self.trace, True
        return arr

    def tick(self):
        self.budget -= 1
        if self.budget < 0:
            raise LoaderError('step budget exceeded (loader does not terminate?)')

    # -- functions ---------------------------------------------------------

    def hoist(self, body_nodes, env):
        for stmt_node in body_nodes:
            for d in walk(stmt_node, kinds={'VariableDeclarator'}, enter_functions=False):
                if d.get('id').kind == 'Identifier':
                    env.vars.setdefault(d.get('id').attr('name'), UNDEF)
            if stmt_node.kind == 'FunctionDeclaration':
                env.vars[_name(stmt_node.get('id'))] = Closure(stmt_node, env)

    def call(self, fn, args):
        if isinstance(fn, Native):
            return fn.fn(*args)
        if not isinstance(fn, Closure):
            raise LoaderError(f'call of a non-function ({fn!r})')
        node = fn.node
        env = Env(fn.env)
        for i, p in enumerate(node.children_in('params')):
            if p.kind != 'Identifier':
                raise LoaderError('unsupported parameter pattern in loader helper')
            env.vars[p.attr('name')] = args[i] if i < len(args) else UNDEF
        body = node.get('body')
        if body.kind != 'BlockStatement':
            return self.ev(body, env)
        self.hoist(body.children_in('body'), env)
        try:
            self.run(body.children_in('body'), env)
        except _Return as r:
            return r.value
        return UNDEF

    # -- statements --------------------------------------------------------

    def run(self, statements, env):
        for s in statements:
            self.stmt(s, env)

    def stmt(self, s, env):
        self.tick()
        k = s.kind
        if k == 'ExpressionStatement':
            self.ev(s.get('expression'), env)
        elif k == 'VariableDeclaration':
            for d in s.children_in('declarations'):
                if d.get('init') is not None:
                    self.assign(d.get('id'), self.ev(d.get('init'), env), env)
        elif k == 'BlockStatement':
            self.run(s.children_in('body'), env)
        elif k == 'IfStatement':
            if V.truthiness(self.ev(s.get('test'), env)) is True:
                self.stmt(s.get('consequent'), env)
            elif s.get('alternate') is not None:
                self.stmt(s.get('alternate'), env)
        elif k in ('WhileStatement', 'DoWhileStatement', 'ForStatement'):
            self.loop(s, env)
        elif k == 'SwitchStatement':
            self.switch(s, env)
        elif k == 'ReturnStatement':
            raise _Return(self.ev(s.get('argument'), env) if s.get('argument') is not None else UNDEF)
        elif k == 'BreakStatement':
            raise _Break()
        elif k == 'ContinueStatement':
            raise _Continue()
        elif k == 'ThrowStatement':
            raise JSThrow(self.ev(s.get('argument'), env))
        elif k in ('EmptyStatement', 'FunctionDeclaration'):
            pass
        else:
            raise LoaderError(f'unsupported statement {k} in loader')

    def loop(self, s, env):
        k = s.kind
        if k == 'ForStatement' and s.get('init') is not None:
            init = s.get('init')
            self.stmt(init, env) if init.is_statement else self.ev(init, env)
        first = k == 'DoWhileStatement'
        while True:
            self.tick()
            if not first:
                test = s.get('test')
                if test is not None and V.truthiness(self.ev(test, env)) is not True:
                    return
            first = False
            try:
                self.stmt(s.get('body'), env)
            except _Break:
                return
            except _Continue:
                pass
            if k == 'ForStatement' and s.get('update') is not None:
                self.ev(s.get('update'), env)

    def switch(self, s, env):
        value = self.ev(s.get('discriminant'), env)
        cases = s.children_in('cases')
        start = None
        for i, c in enumerate(cases):
            if c.get('test') is not None and V.strict_equals(value, self.ev(c.get('test'), env)) is True:
                start = i
                break
        if start is None:
            start = next((i for i, c in enumerate(cases) if c.get('test') is None), None)
        if start is None:
            return
        try:
            for c in cases[start:]:
                self.run(c.children_in('consequent'), env)
        except _Break:
            pass

    # -- expressions -------------------------------------------------------

    def lookup(self, name, env):
        scope = env.find(name)
        if scope is None:
            raise LoaderError(f'unresolved identifier {name!r} in loader')
        return scope.vars[name]

    def assign(self, target, value, env):
        if target.kind == 'Identifier':
            scope = env.find(target.attr('name')) or self.global_env
            scope.vars[target.attr('name')] = value
        elif target.kind == 'MemberExpression':
            obj = self.ev(target.get('object'), env)
            key = _key(self.ev(target.get('property'), env)) if target.attr('computed', False) \
                else target.get('property').attr('name')
            self.set_prop(obj, key, value)
        else:
            raise LoaderError(f'unsupported assignment target {target.kind}')

    def get_prop(self, obj, key):
        key = _key(key)
        if isinstance(obj, str):
            if key == 'length':
                return float(len(obj))
            if isinstance(key, int):
                return obj[key] if 0 <= key < len(obj) else UNDEF
            if key == 'charCodeAt':
                return Native(lambda i=0.0: float(ord(obj[int(i)])) if 0 <= int(i) < len(obj) else math.nan)
        elif isinstance(obj, list):
            if key == 'length':
                return float(len(obj))
            if isinstance(key, int):
                return obj[key] if 0 <= key < len(obj) else UNDEF
            if key == 'push':
                return Native(lambda *a: (obj.extend(a), float(len(obj)))[1])
        elif isinstance(obj, TypedArray):
            if key == 'length':
                return float(obj.length)
            if isinstance(key, int):
                return obj.get(key)
        elif isinstance(obj, dict):
            if key in obj:
                return obj[key]
            return UNDEF
        elif isinstance(obj, _GlobalStub):
            return obj.get(key)
        raise LoaderError(f'unsupported property read {key!r} on {type(obj).__name__}')

    def set_prop(self, obj, key, value):
        key = _key(key)
        if isinstance(obj, TypedArray) and isinstance(key, int):
            obj.set(key, value)
        elif isinstance(obj, list) and isinstance(key, int):
            while len(obj) <= key:
                obj.append(UNDEF)
            obj[key] = value
        elif isinstance(obj, dict):
            obj[key] = value
        else:
            raise LoaderError(f'unsupported property write {key!r} on {type(obj).__name__}')

    def binary(self, op, a, b):
        result = V.binary(op, a, b)
        if result is V.UNKNOWN:
            raise LoaderError(f'unsupported operands for {op!r}: {a!r}, {b!r}')
        return result

    def ev(self, n, env):
        self.tick()
        k = n.kind
        if k == 'Literal':
            return n.attr('value')
        if k == 'Identifier':
            return UNDEF if n.attr('name') == 'undefined' else self.lookup(n.attr('name'), env)
        if k in ('BinaryExpression',):
            return self.binary(n.attr('operator'), self.ev(n.get('left'), env), self.ev(n.get('right'), env))
        if k == 'LogicalExpression':
            left, op = self.ev(n.get('left'), env), n.attr('operator')
            truth = V.truthiness(left)
            if op == '&&':
                return self.ev(n.get('right'), env) if truth is True else left
            if op == '||':
                return left if truth is True else self.ev(n.get('right'), env)
            raise LoaderError(f'unsupported logical operator {op}')
        if k == 'UnaryExpression':
            result = V.unary(n.attr('operator'), self.ev(n.get('argument'), env))
            if result is V.UNKNOWN:
                raise LoaderError(f'unsupported unary {n.attr("operator")}')
            return result
        if k == 'ConditionalExpression':
            branch = 'consequent' if V.truthiness(self.ev(n.get('test'), env)) is True else 'alternate'
            return self.ev(n.get(branch), env)
        if k == 'AssignmentExpression':
            op = n.attr('operator', '=')
            value = self.ev(n.get('right'), env)
            if op != '=':
                value = self.binary(op[:-1], self.ev(n.get('left'), env), value)
            self.assign(n.get('left'), value, env)
            return value
        if k == 'UpdateExpression':
            old = V.to_number(self.ev(n.get('argument'), env))
            new = old + (1.0 if n.attr('operator') == '++' else -1.0)
            self.assign(n.get('argument'), new, env)
            return new if n.attr('prefix', False) else old
        if k == 'SequenceExpression':
            result = UNDEF
            for e in n.children_in('expressions'):
                result = self.ev(e, env)
            return result
        if k == 'MemberExpression':
            obj = self.ev(n.get('object'), env)
            key = self.ev(n.get('property'), env) if n.attr('computed', False) \
                else n.get('property').attr('name')
            return self.get_prop(obj, key)
        if k == 'CallExpression':
            callee = n.get('callee')
            args = [self.ev(a, env) for a in n.children_in('arguments')]
            if callee.kind == 'MemberExpression':
                obj = self.ev(callee.get('object'), env)
                key = self.ev(callee.get('property'), env) if callee.attr('computed', False) \
                    else callee.get('property').attr('name')
                return self.call(self.get_prop(obj, key), args)
            return self.call(self.ev(callee, env), args)
        if k == 'NewExpression':
            ctor = self.ev(n.get('callee'), env)
            args = [self.ev(a, env) for a in n.children_in('arguments')]
            if not isinstance(ctor, NativeCtor):
                raise LoaderError('`new` on an unsupported constructor in loader')
            return ctor.construct(*args)
        if k == 'ArrayExpression':
            return [self.ev(e, env) for e in n.children_in('elements')]
        if k == 'ObjectExpression':
            out = {}
            for p in n.children_in('properties'):
                kn = p.get('key')
                out[kn.attr('name') if kn.kind == 'Identifier' else _key(kn.attr('value'))] = self.ev(p.get('value'), env)
            return out
        if k in FUNCTION_KINDS:
            return Closure(n, env)
        raise LoaderError(f'unsupported expression {k} in loader')


class _GlobalStub:
    """`_global`: constructors we model, an inert opaque value for anything else."""

    def __init__(self, known):
        self.known = known

    def get(self, key):
        return self.known.get(key, V.Opaque(f'_global.{key}'))


# ---------------------------------------------------------------------------
# Locating things in the sample
# ---------------------------------------------------------------------------

def _contains(root, target):
    return any(n is target for n in walk(root))


def _written_values(binding):
    """Initialiser / assigned expressions of a variable."""
    for w in binding.writes:
        parent = w.parent
        if parent is not None and parent.kind == 'AssignmentExpression' and parent.get('left') is w:
            yield parent.get('right')
        elif parent is not None and parent.kind == 'VariableDeclarator' and parent.get('id') is w:
            yield parent.get('init')


def find_payloads(session, interp):
    """[(call node, payload str)] for calls of the factory with a literal payload."""
    outer = interp.outer
    found = []
    for call in walk(session.tree.root, kinds={'CallExpression'}):
        if _contains(outer, call):
            continue
        callee = call.get('callee')
        if callee.kind != 'Identifier':
            continue
        b = session.binding_of(callee)
        if b is None or not any(v is not None and _contains(v, outer) for v in _written_values(b)):
            continue
        args = call.children_in('arguments')
        if not args:
            continue
        payload = _literal_string(args[0])
        if payload is not None:
            found.append((call, payload))
    return found


def _literal_string(n):
    if n.kind == 'Literal' and isinstance(n.attr('value'), str):
        return n.attr('value')
    if n.kind == 'CallExpression' and _name(n.get('callee')) in ('decodeURIComponent', 'decodeURI', 'unescape'):
        inner = n.children_in('arguments')
        if len(inner) == 1 and inner[0].kind == 'Literal' and isinstance(inner[0].attr('value'), str):
            return unquote(inner[0].attr('value'), errors='strict')
    return None


def _jump_base(interp):
    for fn in interp.func.get('body').children_in('body'):
        h = interp.helpers.get(_name(fn.get('id'))) if fn.kind == 'FunctionDeclaration' else None
        if h is not None and h.kind == 'jump':
            params = [_name(p) for p in fn.children_in('params')]
            for a in walk(fn, kinds={'AssignmentExpression'}):
                if _name(a.get('left')) == interp.pc_name and a.get('right').kind == 'BinaryExpression':
                    for side in (a.get('right').get('left'), a.get('right').get('right')):
                        if _name(side) and _name(side) not in params:
                            return _name(side)
    return None


def _exception_table_name(session, interp):
    for catch in walk(interp.func, kinds={'CatchClause'}):
        for call in walk(catch, kinds={'CallExpression'}):
            callee = call.get('callee')
            b = session.binding_of(callee) if callee.kind == 'Identifier' else None
            fn = b.declaration if b is not None else None
            if fn is None or fn.kind != 'FunctionDeclaration':
                continue
            for m in walk(fn, kinds={'MemberExpression'}):
                prop = m.get('property')
                if m.attr('computed', False) and prop.kind == 'BinaryExpression' and prop.attr('operator') == '-' \
                        and prop.get('right').kind == 'Literal' and prop.get('right').attr('value') in (1.0, 2.0, 3.0):
                    return _name(m.get('object'))
    return None


class Layout:
    """Everything about a build's loader that is not the payload itself."""

    def __init__(self, session, interp, consts_name):
        self.session, self.interp = session, interp
        self.consts_name = consts_name
        self.base_name = _jump_base(interp)
        self.exceptions_name = _exception_table_name(session, interp)
        body = interp.outer.get('body').children_in('body')
        self.index = next(i for i, s in enumerate(body) if _contains(s, interp.func))
        self.loader, self.post = body[:self.index], body[self.index + 1:]
        self._key_name = self._start = None
        self._interp_call()
        self.flags_name = self.flags_expected = self.name_var = self.arity_var = None
        self._epilogue()

    def _interp_call(self):
            fb = None
            for ident in walk(self.session.tree.root, kinds={'Identifier'}):
                b = self.session.binding_of(ident)
                if b is not None and b.declaration is not None and ident.parent is not None \
                        and ident.parent.kind == 'CallExpression' and ident.parent.get('callee') is ident \
                        and any(v is self.interp.func for v in _written_values(b)):
                    fb = ident.parent
                    break
            if fb is None:
                raise LoaderError('could not find the call that starts the interpreter')

            self.interp_call_node = fb   # <--- Save the AST node here

            args = fb.children_in('arguments')
            self._start = args[2].attr('value') if args[2].kind == 'Literal' else 0.0
            self._key_name = _name(args[3])

    def _epilogue(self):
        for s in self.post:
            for n in walk(s):
                if n.kind == 'IfStatement' and n.get('test').kind == 'BinaryExpression' \
                        and n.get('test').attr('operator') == '===':
                    left, right = n.get('test').get('left'), n.get('test').get('right')
                    if left.kind == 'Identifier' and right.kind == 'Literal' and isinstance(right.attr('value'), float):
                        self.flags_name, self.flags_expected = _name(left), right.attr('value')
                if n.kind == 'CallExpression' and n.get('callee').kind == 'MemberExpression' \
                        and n.get('callee').get('property').attr('name') == 'defineProperty':
                    args = n.children_in('arguments')
                    if len(args) == 3 and args[1].kind == 'Literal' and args[2].kind == 'ObjectExpression':
                        value = next((p.get('value') for p in args[2].children_in('properties')
                                      if _name(p.get('key')) == 'value'), None)
                        if args[1].attr('value') == 'name' and value is not None and value.kind == 'MemberExpression':
                            self.name_var = _name(value.get('property'))
                        elif args[1].attr('value') == 'length' and value is not None:
                            self.arity_var = _name(value)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

class Decoded:
    def __init__(self):
        self.payload = None
        self.data = b''
        self.key = self.code_base = None
        self.flags = self.flags_expected = None
        self.constants = []
        self.exceptions = []          # (start, end, handler, key); lookup scans from the last entry backwards
        self.name = self.arity = None
        self.reads = []               # contiguous (start, end) runs of payload bytes the loader consumed
        self.scalars = {}
        self.env = self.machine = None

    @property
    def code(self):
        return self.data[self.code_base:]

    @property
    def valid(self):
        """Loader consumed exactly [0, code_base) and the guard flag matches."""
        return (self.flags == self.flags_expected and self.reads == [(0, self.code_base)]
                and self.code_base <= len(self.data))


def _runs(offsets):
    runs = []
    for o in sorted(set(offsets)):
        if runs and o == runs[-1][1]:
            runs[-1] = (runs[-1][0], o + 1)
        else:
            runs.append((o, o + 1))
    return runs


def decode_payload(layout, payload):
    session, interp = layout.session, layout.interp
    machine = Machine(len(payload))
    outer = interp.outer
    env = Env(machine.global_env)
    params = [p.attr('name') for p in outer.children_in('params')]
    for i, name in enumerate(params):
        env.vars[name] = payload if i == 0 else UNDEF
    machine.hoist(layout.loader, env)
    try:
        machine.run([s for s in layout.loader if s.kind != 'FunctionDeclaration'], env)
    except JSThrow as exc:
        raise LoaderError('loader rejected the payload (threw)') from exc
    except _Return:
        raise LoaderError('loader returned early')

    out = Decoded()
    out.payload = payload
    arr = env.vars[interp.bytes_name]
    out.data = bytes(arr.buf.data)
    out.key = int(env.vars[layout._key_name])
    out.code_base = int(env.vars[layout.base_name])
    out.constants = [_clean(v) for v in env.vars[layout.consts_name]]
    exc = [int(x) for x in env.vars[layout.exceptions_name]] if layout.exceptions_name else []
    out.exceptions = [tuple(exc[i:i + 4]) for i in range(0, len(exc), 4)]
    if layout.flags_name:
        out.flags, out.flags_expected = int(env.vars[layout.flags_name]), int(layout.flags_expected)
    if layout.arity_var:
        out.arity = int(env.vars[layout.arity_var])
    if layout.name_var:
        idx = int(env.vars[layout.name_var])
        out.name = out.constants[idx] if 0 <= idx < len(out.constants) else None
    out.reads = _runs(machine.trace)
    out.scalars = {k: v for k, v in env.vars.items() if isinstance(v, float)}
    out.env, out.machine = env, machine          # reused by the disassembler to run the build's own readers
    return out


def _clean(v):
    return v


def decode_children(layout, decoded, seen=None, depth=0):
    """{constant index: Decoded} for string constants that are themselves valid payloads."""
    seen = seen if seen is not None else set()
    out = {}
    if depth > 8:
        return out
    for i, c in enumerate(decoded.constants):
        if not isinstance(c, str) or len(c) < 8 or c in seen:
            continue
        seen.add(c)
        try:
            child = decode_payload(layout, c)
        except (LoaderError, IndexError, KeyError, ValueError, TypeError):
            continue
        if child.valid:
            out[i] = child
    return out


def load(path):
    source = open(path, encoding='utf-8').read()
    try:
        return Session.parse(source)
    except ImportError:
        import esprima
        return Session.from_ast(esprima.parseScript(source))


def build_layout(session):
    analysis = analyse(session)
    consts = next(b.name for b, r in analysis.roles.items() if r == CONSTS)
    return Layout(session, analysis.interp, consts), analysis


def show(layout, decoded, indent='', limit=60, depth=0, seen=None):
    seen = seen if seen is not None else set()

    def r(v):
        text = V.number_to_string(v) if isinstance(v, float) else repr(v)
        return text if len(text) <= limit else text[:limit] + f'... ({len(text)} chars)'

    pad = indent
    print(f'{pad}function {decoded.name!r}  arity={decoded.arity}  flags={decoded.flags} '
          f'(expect {decoded.flags_expected})  key=0x{decoded.key:02x}')
    print(f'{pad}  header bytes [0, {decoded.code_base}) read as {decoded.reads}   '
          f'code: {len(decoded.code)} bytes   valid={decoded.valid}')
    print(f'{pad}  exception table: {decoded.exceptions or "-"}')
    print(f'{pad}  constants ({len(decoded.constants)}):')
    for i, c in enumerate(decoded.constants):
        print(f'{pad}    [{i}] {r(c)}')
    if depth < 6:
        for i, child in decode_children(layout, decoded, seen).items():
            print(f'{pad}  child via constant [{i}]:')
            show(layout, child, pad + '    ', limit, depth + 1, seen)


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    session = load(argv[1])
    layout, analysis = build_layout(session)
    it = layout.interp
    print(f'factory {it.outer_name}: loader = {len(layout.loader)} statements before the interpreter')
    print(f'bytes={it.bytes_name} consts={layout.consts_name} exceptions={layout.exceptions_name} '
          f'key={layout._key_name} code_base={layout.base_name} flags={layout.flags_name}=={layout.flags_expected} '
          f'name=consts[{layout.name_var}] arity={layout.arity_var}')
    payloads = find_payloads(session, it)
    if not payloads:
        print('no call of the factory with a literal payload found')
        return 2
    for call, payload in payloads:
        print(f'\npayload: {len(payload)} chars')
        show(layout, decode_payload(layout, payload))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
