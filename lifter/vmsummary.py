"""
vmsummary -- derive per-opcode semantic summaries from a stack-VM interpreter.

Instead of a table of known instructions, this reads the interpreter's own
`switch` and works out what each handler *does*, symbolically:

    pops N values, reads these operands, and (conditionally) performs these
    effects, then pushes these values.

The result is a *template* per opcode, written in jsir nodes with placeholder
identifiers, so a lifter can instantiate it against a symbolic stack without
knowing anything about the specific build:

    $in0, $in1, ...   values popped from (or peeked at on) the stack, top first
    $b0, $v0, $w0     operands: 8-bit, variable-length, and 32-bit, by kind
    t0, t1, ...       results of impure operations, bound once, in order
    $const(i) $arg(i) $local(i) $scope(d,i) $global(name) $ext_get(i) ...
                      role-resolved environment access (see ROLES)
    $call(fn, this, ...args)   $new(...)   $jump(target, key)   $setkey(k)
    $return(x)  $throw(x)  $poke(addr, v)   $child_vm(code, scope, ...)

Pipeline
    1. find the dispatch loop: the big numeric `switch` fed by a zero-arg fetch
    2. classify the interpreter's helper functions by what their bodies do
       (byte / varint / 32-bit operand readers, jump, set-key, poke)
    3. infer the *role* of each free variable (stack, locals, constants, args,
       scope chain, globals, externals) from how the handlers use it
    4. symbolically execute every case body into a Summary

Everything here is name-independent, so the summaries of two builds can be
compared directly (see `match_builds`), which recovers the opcode renumbering
between builds without any hand-written table.

Soundness notes
  * Straight-line handlers plus `if (cond) <jump|setkey>` are modelled exactly.
    Anything else is recorded in Summary.notes instead of being guessed at.
  * Impure operations (calls, `new`, member loads, allocations, world reads) are
    bound to temporaries in evaluation order. Pure arithmetic is left inline.
    Inlining temporaries is deliberately left to a later cleanup pass.
  * Operators are inlined only when they provably cannot run user code
    (=== / !== / ! / typeof, or arithmetic on primitives). Anything that may
    coerce an object (`a + b`, `a < b`, `-a`, `k in o`) is bound to a temporary
    so it stays ordered with calls, loads and stores; vmcheck found that the
    "operators are pure" shortcut reorders observable behaviour.
"""

import json
import re
import sys
from collections import Counter, defaultdict

from jsir import Session, walk, node as make, clone, values as V
from jsir.schema import FUNCTION_KINDS

STACK, LOCALS, CONSTS, ARGS, SCOPE, GLOBALS, EXTERNALS = (
    'STACK', 'LOCALS', 'CONSTS', 'ARGS', 'SCOPE', 'GLOBALS', 'EXTERNALS')

OPERAND_RE = re.compile(r'^\$([bvw])(\d+)$')
KIND_LETTER = {'byte': 'b', 'varint': 'v', 'word': 'w'}


# ---------------------------------------------------------------------------
# Node helpers
# ---------------------------------------------------------------------------

def ident(name):
    return make('Identifier', name=name)


def num(value):
    return make('Literal', value=float(value))


def intr(name, *args):
    return make('CallExpression', callee=ident(name), arguments=list(args))


def stmt(expression):
    return make('ExpressionStatement', expression=expression)


def let(name, init):
    return make('VariableDeclaration', kind='let', declarations=[
        make('VariableDeclarator', id=ident(name), init=init)])


def _is_primitive(e):
    """True if the expression provably yields a primitive, so operators applied
    to it cannot run valueOf/toString or hit a Proxy trap."""
    k = e.kind
    if k == 'Literal':
        return True
    if k == 'Identifier':
        return bool(OPERAND_RE.match(e.attr('name'))) or e.attr('name') == 'undefined'
    if k == 'UnaryExpression':
        return e.attr('operator') in ('!', 'typeof', 'void') or _is_primitive(e.get('argument'))
    if k == 'BinaryExpression':
        return e.attr('operator') in ('===', '!==') or (
            e.attr('operator') not in ('in', 'instanceof')
            and _is_primitive(e.get('left')) and _is_primitive(e.get('right')))
    return False


def _is_num_lit(n):
    return n is not None and n.kind == 'Literal' and isinstance(n.attr('value'), float)


def _name(n):
    return n.attr('name') if n is not None and n.kind == 'Identifier' else None


# ---------------------------------------------------------------------------
# Printer (small, dependency-free; used for display and for fingerprints)
# ---------------------------------------------------------------------------

def show(n, mask=frozenset(), hook=None):
    """hook(node) -> str | None lets callers substitute operand/constant values."""
    if hook is not None:
        text = hook(n)
        if text is not None:
            return text
    k = n.kind
    if k == 'Identifier':
        name = n.attr('name')
        return '$free' if name in mask else name
    if k == 'Literal':
        v = n.attr('value')
        if v is V.NULL:
            return 'null'
        if isinstance(v, bool):
            return 'true' if v else 'false'
        if isinstance(v, float):
            return V.number_to_string(v)
        return json.dumps(v)
    if k == 'UnaryExpression':
        op = n.attr('operator')
        sep = ' ' if op.isalpha() else ''
        return f'{op}{sep}{show(n.get("argument"), mask, hook)}'
    if k in ('BinaryExpression', 'LogicalExpression'):
        return f'({show(n.get("left"), mask, hook)} {n.attr("operator")} {show(n.get("right"), mask, hook)})'
    if k == 'AssignmentExpression':
        return f'{show(n.get("left"), mask, hook)} {n.attr("operator", "=")} {show(n.get("right"), mask, hook)}'
    if k == 'MemberExpression':
        obj = show(n.get('object'), mask, hook)
        if n.attr('computed', False):
            return f'{obj}[{show(n.get("property"), mask, hook)}]'
        return f'{obj}.{n.get("property").attr("name")}'
    if k == 'CallExpression':
        args = ', '.join(show(a, mask, hook) for a in n.children_in('arguments'))
        return f'{show(n.get("callee"), mask, hook)}({args})'
    if k == 'NewExpression':
        args = ', '.join(show(a, mask, hook) for a in n.children_in('arguments'))
        return f'new {show(n.get("callee"), mask, hook)}({args})'
    if k == 'ArrayExpression':
        return '[' + ', '.join(show(e, mask, hook) for e in n.children_in('elements')) + ']'
    if k == 'ObjectExpression':
        return '{}'
    if k == 'SpreadElement':
        return '...' + show(n.get('argument'), mask, hook)
    if k == 'ConditionalExpression':
        return (f'({show(n.get("test"), mask, hook)} ? {show(n.get("consequent"), mask, hook)} '
                f': {show(n.get("alternate"), mask, hook)})')
    return f'<{k}>'


def show_stmt(s, indent='  ', mask=frozenset(), hook=None):
    k = s.kind
    if k == 'ExpressionStatement':
        return [indent + show(s.get('expression'), mask, hook) + ';']
    if k == 'VariableDeclaration':
        d = s.children_in('declarations')[0]
        return [f'{indent}let {d.get("id").attr("name")} = {show(d.get("init"), mask, hook)};']
    if k == 'IfStatement':
        lines = [f'{indent}if {show(s.get("test"), mask, hook)} {{']
        for inner in s.get('consequent').children_in('body'):
            lines += show_stmt(inner, indent + '  ', mask, hook)
        alt = s.get('alternate')
        if alt is not None:
            lines.append(f'{indent}}} else {{')
            for inner in alt.children_in('body'):
                lines += show_stmt(inner, indent + '  ', mask, hook)
        lines.append(f'{indent}}}')
        return lines
    return [f'{indent}<{k}>']


# ---------------------------------------------------------------------------
# Step 1-2: locate the dispatch loop, classify helpers
# ---------------------------------------------------------------------------

class Helper:
    def __init__(self, name, kind, **extra):
        self.name = name
        self.kind = kind          # byte|varint|word|jump|setkey|poke|other
        self.extra = extra

    def __repr__(self):
        return f'{self.name}:{self.kind}{self.extra or ""}'


def _byte_reads(fn):
    """Occurrences of `BYTES[PC++] ^ KEY` -- the shape of an operand read."""
    reads = []
    for b in walk(fn, kinds={'BinaryExpression'}):
        if b.attr('operator') != '^':
            continue
        left, right = b.get('left'), b.get('right')
        if (left.kind == 'MemberExpression' and left.attr('computed', False)
                and left.get('property').kind == 'UpdateExpression'
                and left.get('object').kind == 'Identifier'
                and right.kind == 'Identifier'):
            reads.append((_name(left.get('object')),
                          _name(left.get('property').get('argument')),
                          _name(right)))
    return reads


class Interpreter:
    """What was found about one VM interpreter."""

    def __init__(self, session):
        self.session = session
        self._locate_dispatch()
        self._classify_helpers()
        self._find_stack()

    # -- bindings ---------------------------------------------------------

    def binding(self, identifier):
        return self.session.binding_of(identifier)

    def is_local(self, binding):
        """A handler-private variable (declared with var in the interpreter)."""
        return binding.kind == 'var' and binding.scope.function is self.func

    def is_param(self, binding):
        return binding.kind == 'param' and binding.scope.function is self.func

    # -- dispatch ---------------------------------------------------------

    def _locate_dispatch(self):
        best = None
        for sw in walk(self.session.tree.root, kinds={'SwitchStatement'}):
            count = sum(1 for c in sw.children_in('cases') if _is_num_lit(c.get('test')))
            if count >= 8 and (best is None or count > best[0]):
                best = (count, sw)
        if best is None:
            raise ValueError('no opcode switch found (needs >= 8 numeric cases)')
        self.switch = best[1]
        self.func = self.switch.enclosing(*FUNCTION_KINDS)
        self.outer = self.func.enclosing(*FUNCTION_KINDS)
        outer_id = self.outer.get('id') if self.outer is not None else None
        self.outer_name = _name(outer_id)

        disc = _name(self.switch.get('discriminant'))
        self.fetch_name = None
        for d in walk(self.func, kinds={'VariableDeclarator'}):
            init = d.get('init')
            if _name(d.get('id')) == disc and init is not None and init.kind == 'CallExpression' \
                    and _name(init.get('callee')):
                self.fetch_name = _name(init.get('callee'))
                break

        self.cases = {}
        for c in self.switch.children_in('cases'):
            if _is_num_lit(c.get('test')):
                self.cases[int(c.get('test').attr('value'))] = c

    def _classify_helpers(self):
        self.helpers = {}
        body = self.func.get('body')
        fns = [s for s in body.children_in('body') if s.kind == 'FunctionDeclaration']
        trio = None
        for fn in fns:
            if fn.children_in('params'):
                continue
            reads = _byte_reads(fn)
            if not reads:
                continue
            trio = trio or reads[0]
            has_switch = any(True for _ in walk(fn, kinds={'SwitchStatement'}))
            if has_switch:
                kind = 'varint'
            elif len(reads) == 1:
                kind = 'byte'
            elif len(reads) == 4:
                kind = 'word'
            else:
                kind = 'other'
            self.helpers[_name(fn.get('id'))] = Helper(_name(fn.get('id')), kind)
        self.bytes_name, self.pc_name, self.key_name = trio or (None, None, None)

        for fn in fns:
            fname = _name(fn.get('id'))
            if fname in self.helpers:
                continue
            params = [_name(p) for p in fn.children_in('params')]
            key_param = target_param = None
            poke = False
            for a in walk(fn, kinds={'AssignmentExpression'}):
                left, right = a.get('left'), a.get('right')
                if _name(left) == self.key_name and _name(right) in params:
                    key_param = params.index(_name(right))
                elif _name(left) == self.pc_name and right.kind == 'BinaryExpression':
                    for side in (right.get('left'), right.get('right')):
                        if _name(side) in params:
                            target_param = params.index(_name(side))
                elif left.kind == 'MemberExpression' and _name(left.get('object')) == self.bytes_name:
                    poke = True
            if target_param is not None and key_param is not None:
                self.helpers[fname] = Helper(fname, 'jump', target=target_param, key=key_param)
            elif key_param is not None and len(params) == 1:
                self.helpers[fname] = Helper(fname, 'setkey')
            elif poke and len(params) == 2:
                self.helpers[fname] = Helper(fname, 'poke')
            else:
                self.helpers[fname] = Helper(fname, 'other')

    def _find_stack(self):
        counts = Counter()
        for call in walk(self.switch, kinds={'CallExpression'}):
            callee = call.get('callee')
            if callee.kind == 'MemberExpression' and not callee.attr('computed', False) \
                    and callee.get('property').attr('name') in ('push', 'pop', 'splice') \
                    and callee.get('object').kind == 'Identifier':
                b = self.binding(callee.get('object'))
                if b is not None and not self.is_local(b):
                    counts[b] += 1
        if not counts:
            raise ValueError('could not identify the operand stack')
        self.stack = counts.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Step 3-4: symbolic execution of one handler
# ---------------------------------------------------------------------------

class Facts:
    """Usage evidence gathered during a role-less discovery pass."""

    def __init__(self):
        self.tags = defaultdict(Counter)
        self.ld_from = defaultdict(Counter)


class Summary:
    def __init__(self, opcode):
        self.opcode = opcode
        self.layout = []          # operand names in read order: ['b0', 'w0', ...]
        self.pops = 0
        self.depth = 0            # minimum stack height required
        self.explicit = False     # stack ops are in `body` rather than pops/pushes
        self.body = []
        self.pushes = []
        self.notes = []
        self.free = set()         # unresolved environment names

    @property
    def layout_kinds(self):
        return ''.join(name[0] for name in self.layout)

    def lines(self, canon=False):
        mask = frozenset(self.free) if canon else frozenset()
        out = []
        for s in self.body:
            out += show_stmt(s, '  ', mask)
        for p in self.pushes:
            out.append(f'  push {show(p, mask)};')
        return out

    def fingerprint(self):
        head = f'pops={self.pops} depth={self.depth} explicit={self.explicit}'
        return head + '\n' + '\n'.join(self.lines(canon=True))

    def render(self):
        stack = 'dynamic' if self.explicit else f'pops {self.pops}, pushes {len(self.pushes)}'
        head = (f'op {self.opcode:<3} operands [{", ".join("$" + n for n in self.layout) or "-"}]'
                f'  stack: {stack}')
        if self.depth > self.pops and not self.explicit:
            head += f' (peeks to depth {self.depth})'
        body = self.lines() or ['  (no effect)']
        out = [head] + body
        for note in self.notes:
            out.append(f'  ! {note}')
        return '\n'.join(out)


class HandlerEval:
    def __init__(self, interp, roles, facts, summary):
        self.it = interp
        self.roles = roles
        self.facts = facts            # None once roles are known
        self.sum = summary
        self.env = {}                 # Binding -> expression node
        self.stmts = []
        self.out = []
        self.consumed = 0
        self.max_depth = 0
        self.kind_count = Counter()
        self.ntemp = 0
        self.tdef = {}                # temp name -> Binding it was loaded from
        self.explicit = False
        self.done = False
        self.in_child = False
        self.const_binding = next((b for b, r in roles.items() if r == CONSTS), None)

    # -- small utilities -------------------------------------------------

    def note(self, text):
        if text not in self.sum.notes:
            self.sum.notes.append(text)

    def tag(self, binding, name):
        if self.facts is not None:
            self.facts.tags[binding][name] += 1

    def bind(self, expression, source=None):
        name = f't{self.ntemp}'
        self.ntemp += 1
        self.stmts.append(let(name, expression))
        if source is not None:
            self.tdef[name] = source
        return ident(name)

    def free(self, name):
        self.sum.free.add(name)
        return ident(name)

    # -- stack model -----------------------------------------------------

    def pop(self):
        if self.explicit:
            return self.bind(intr('$pop'))
        if self.out:
            return self.out.pop()
        depth = self.consumed
        self.consumed += 1
        self.max_depth = max(self.max_depth, depth + 1)
        return ident(f'$in{depth}')

    def push(self, value):
        if self.explicit:
            self.stmts.append(stmt(intr('$push', value)))
        else:
            self.out.append(value)

    def peek(self, k):
        if self.explicit:
            return self.bind(intr('$peek', num(k)))
        if k < len(self.out):
            return clone(self.out[-1 - k])
        depth = self.consumed + (k - len(self.out))
        self.max_depth = max(self.max_depth, depth + 1)
        return ident(f'$in{depth}')

    def go_explicit(self):
        """Dynamic stack arity: fall back to stepwise $push/$pop statements."""
        if not self.explicit:
            for value in self.out:
                self.stmts.append(stmt(intr('$push', value)))
            self.out = []
            self.explicit = True

    def stack_peek_depth(self, member):
        """STACK[STACK.length - a - b ...] -> depth, or None."""
        if not member.attr('computed', False):
            return None
        if self.it.binding(member.get('object')) is not self.it.stack \
                if member.get('object').kind == 'Identifier' else True:
            return None
        total, cur = 0, member.get('property')
        while cur.kind == 'BinaryExpression' and cur.attr('operator') == '-':
            if not _is_num_lit(cur.get('right')):
                return None
            total += int(cur.get('right').attr('value'))
            cur = cur.get('left')
        if cur.kind == 'MemberExpression' and not cur.attr('computed', False) \
                and cur.get('property').attr('name') == 'length' \
                and self.it.binding(cur.get('object')) is self.it.stack:
            return total - 1
        return None

    # -- expressions -----------------------------------------------------

    def ev(self, n, discard=False):
        k = n.kind
        if k == 'Literal':
            return clone(n)
        if k == 'Identifier':
            return self.ev_ident(n)
        if k == 'MemberExpression':
            return self.ev_member(n)
        if k == 'CallExpression':
            return self.ev_call(n, discard)
        if k == 'NewExpression':
            callee = self.ev(n.get('callee'))
            args = [self.ev(a) for a in n.children_in('arguments')]
            return self.bind(make('NewExpression', callee=callee, arguments=args))
        if k == 'UnaryExpression':
            op = n.attr('operator')
            arg = self.ev(n.get('argument'))
            expr = make('UnaryExpression', operator=op, prefix=True, argument=arg)
            if op == 'delete':
                self.note('delete operator')
                return self.bind(expr)
            if op in ('-', '+', '~') and not _is_primitive(arg):
                return self.bind(expr)            # numeric coercion may run user code
            return expr
        if k in ('BinaryExpression', 'LogicalExpression'):
            left = self.ev(n.get('left'))
            mark = len(self.stmts)
            right = self.ev(n.get('right'))
            if k == 'LogicalExpression' and len(self.stmts) != mark:
                self.note('impure right operand of a short-circuit operator')
            expr = make(k, operator=n.attr('operator'), left=left, right=right)
            if k == 'BinaryExpression' and not _is_primitive(expr):
                return self.bind(expr)            # coercion / traps: keep in evaluation order
            return expr
        if k == 'ConditionalExpression':
            test = self.ev(n.get('test'))
            mark = len(self.stmts)
            a, b = self.ev(n.get('consequent')), self.ev(n.get('alternate'))
            if len(self.stmts) != mark:
                self.note('impure arm of a conditional expression')
            return make('ConditionalExpression', test=test, consequent=a, alternate=b)
        if k == 'ArrayExpression':
            return self.ev_array(n)
        if k == 'SpreadElement':
            return make('SpreadElement', argument=self.ev(n.get('argument')))
        if k == 'ObjectExpression':
            if n.children_in('properties'):
                self.note('non-empty object literal')
            return self.bind(make('ObjectExpression', properties=[]))
        if k == 'MetaProperty':
            return ident('$new_target')
        if k in FUNCTION_KINDS:
            self.note('function expression inside a handler')
            return ident('<fn>')
        self.note(f'unmodelled expression {k}')
        return ident('$unsupported')

    def ev_ident(self, n):
        b = self.it.binding(n)
        name = n.attr('name')
        if b is None:
            return ident(name)                       # true global (Symbol, ...)
        if self.it.is_local(b):
            if b in self.env:
                return clone(self.env[b])
            self.note(f'read of unassigned local {name}')
            return ident('undefined')
        role = self.roles.get(b)
        if role == EXTERNALS:
            return ident('$externals')
        return self.free(name)

    def chain(self, n):
        """(binding, [index nodes]) for env[a][b] rooted at a non-local name."""
        idx, cur = [], n
        while cur.kind == 'MemberExpression' and cur.attr('computed', False):
            idx.append(cur.get('property'))
            cur = cur.get('object')
        if cur.kind != 'Identifier' or not idx or len(idx) > 2:
            return None
        b = self.it.binding(cur)
        if b is None or self.it.is_local(b):
            return None
        idx.reverse()
        return b, idx

    def index_kind(self, e):
        if e.kind == 'Identifier':
            if OPERAND_RE.match(e.attr('name')):
                return 'op'
            src = self.tdef.get(e.attr('name'))
            if src is not None:
                return src
        if e.kind == 'CallExpression' and _name(e.get('callee')) == '$const':
            return self.const_binding
        return None

    def access(self, b, exprs, value=None, write=False):
        kinds = [self.index_kind(e) for e in exprs]
        if self.facts is not None:
            if len(exprs) == 2 and all(k == 'op' for k in kinds):
                self.tag(b, 'idx2')
            elif len(exprs) == 1 and kinds[0] == 'op':
                self.tag(b, 'idx1_write' if write else 'idx1_read')
            elif len(exprs) == 1 and kinds[0] not in (None, 'op'):
                self.facts.ld_from[b][kinds[0]] += 1
        role = self.roles.get(b)
        n = len(exprs)
        if not write:
            if role == CONSTS and n == 1:
                return intr('$const', *exprs)
            if role == ARGS and n == 1:
                return self.bind(intr('$arg', *exprs))
            if role == LOCALS and n == 1:
                return self.bind(intr('$local', *exprs))
            if role == SCOPE:
                return self.bind(intr('$scope' if n == 2 else '$scope_row', *exprs))
            if role == GLOBALS and n == 1:
                return self.bind(intr('$global', *exprs))
            expr = ident(b.name)
            for e in exprs:
                expr = make('MemberExpression', object=expr, property=e, computed=True)
            self.sum.free.add(b.name)
            return self.bind(expr, source=b)
        if role == LOCALS and n == 1:
            self.stmts.append(stmt(intr('$set_local', *exprs, value)))
        elif role == SCOPE:
            self.stmts.append(stmt(intr('$set_scope' if n == 2 else '$set_scope_row', *exprs, value)))
        elif role == GLOBALS and n == 1:
            self.stmts.append(stmt(intr('$set_global', *exprs, value)))
        else:
            target = ident(b.name)
            for e in exprs:
                target = make('MemberExpression', object=target, property=e, computed=True)
            self.sum.free.add(b.name)
            self.stmts.append(stmt(make('AssignmentExpression', operator='=', left=target, right=value)))
        return None

    def ev_member(self, n):
        depth = self.stack_peek_depth(n)
        if depth is not None:
            return self.peek(depth)
        found = self.chain(n)
        if found is not None:
            b, idx_nodes = found
            return self.access(b, [self.ev(i) for i in idx_nodes])
        obj = self.ev(n.get('object'))
        if n.attr('computed', False):
            prop = self.ev(n.get('property'))
            expr = make('MemberExpression', object=obj, property=prop, computed=True)
        else:
            expr = make('MemberExpression', object=obj, property=ident(n.get('property').attr('name')),
                        computed=False)
        return self.bind(expr)

    def ev_array(self, n):
        elements = n.children_in('elements')
        if len(elements) == 1 and elements[0].kind == 'SpreadElement' \
                and elements[0].get('argument').kind == 'Identifier':
            b = self.it.binding(elements[0].get('argument'))
            if b is not None and not self.it.is_local(b):
                self.tag(b, 'spread_child' if self.in_child else 'spread_array')
                role = self.roles.get(b)
                if role == SCOPE:
                    return intr('$scope_snapshot')
                if role == ARGS:
                    return self.bind(intr('$args_copy'))
        items = [self.ev(e) for e in elements]
        return self.bind(make('ArrayExpression', elements=items))

    def read_operand(self, kind):
        letter = KIND_LETTER[kind]
        name = f'{letter}{self.kind_count[letter]}'
        self.kind_count[letter] += 1
        self.sum.layout.append(name)
        return ident('$' + name)

    def ev_call(self, n, discard):
        callee, args_ast = n.get('callee'), n.children_in('arguments')

        if callee.kind == 'Identifier':
            name = callee.attr('name')
            helper = self.it.helpers.get(name)
            if helper is not None:
                return self.helper_call(helper, args_ast)
            if name == self.it.outer_name:
                self.in_child = True
                args = [self.ev(a) for a in args_ast]
                self.in_child = False
                return self.bind(intr('$child_vm', *args))
            fn = self.ev(callee)
            args = [self.ev(a) for a in args_ast]
            return self.finish_call(make('CallExpression', callee=fn, arguments=args), discard)

        if callee.kind == 'MemberExpression':
            obj, computed = callee.get('object'), callee.attr('computed', False)
            pname = None if computed else callee.get('property').attr('name')

            # operand-stack operations
            if obj.kind == 'Identifier' and self.it.binding(obj) is self.it.stack:
                if pname == 'pop':
                    return self.pop()
                if pname == 'push':
                    self.push(self.ev(args_ast[0]))
                    return ident('undefined')
                if pname == 'splice':
                    return self.splice(args_ast)

            # CALL_IMPORT: EXTERNALS[i][0]()
            if computed and _is_num_lit(callee.get('property')) \
                    and callee.get('property').attr('value') == 0.0:
                found = self.chain(obj)
                if found is not None and len(found[1]) == 1:
                    b, (idx,) = found
                    self.tag(b, 'ext_get')
                    index = self.ev(idx)
                    if self.roles.get(b) == EXTERNALS:
                        return self.bind(intr('$ext_get', index))

            # fn.call(this, ...args)
            if pname == 'call':
                fn = self.ev(obj)
                args = [self.ev(a) for a in args_ast]
                return self.finish_call(intr('$call', fn, *args), discard)

            # generic method call: evaluate the receiver, keep the method name
            receiver = self.ev(obj)
            if computed:
                method = make('MemberExpression', object=receiver, property=self.ev(callee.get('property')),
                              computed=True)
            else:
                method = make('MemberExpression', object=receiver, property=ident(pname), computed=False)
            args = [self.ev(a) for a in args_ast]
            return self.finish_call(make('CallExpression', callee=method, arguments=args), discard)

        fn = self.ev(callee)
        args = [self.ev(a) for a in args_ast]
        return self.finish_call(make('CallExpression', callee=fn, arguments=args), discard)

    def finish_call(self, expression, discard):
        if discard:
            self.stmts.append(stmt(expression))
            return ident('undefined')
        return self.bind(expression)

    def helper_call(self, helper, args_ast):
        if helper.kind in KIND_LETTER:
            return self.read_operand(helper.kind)
        args = [self.ev(a) for a in args_ast]
        if helper.kind == 'jump':
            self.stmts.append(stmt(intr('$jump', args[helper.extra['target']], args[helper.extra['key']])))
        elif helper.kind == 'setkey':
            self.stmts.append(stmt(intr('$setkey', args[0])))
        elif helper.kind == 'poke':
            self.note('SELF-MODIFYING: writes to the bytecode')
            self.stmts.append(stmt(intr('$poke', *args)))
        else:
            self.note(f'call to unclassified helper {helper.name}')
            self.stmts.append(stmt(intr(helper.name, *args)))
        return ident('undefined')

    def splice(self, args_ast):
        count = self.ev(args_ast[1]) if len(args_ast) > 1 else None
        if count is not None and count.kind == 'Literal':
            items = [self.pop() for _ in range(int(count.attr('value')))]
            items.reverse()
            return self.bind(make('ArrayExpression', elements=items))
        self.go_explicit()
        return self.bind(intr('$popn', count if count is not None else ident('undefined')))

    # -- statements ------------------------------------------------------

    def run(self, statements):
        for s in statements:
            if self.done:
                return
            self.stmt(s)

    def stmt(self, s):
        k = s.kind
        if k == 'BlockStatement':
            self.run(s.children_in('body'))
        elif k == 'VariableDeclaration':
            for d in s.children_in('declarations'):
                if d.get('init') is not None:
                    self.assign_local(d.get('id'), self.ev(d.get('init')))
        elif k == 'ExpressionStatement':
            self.expression_statement(s.get('expression'))
        elif k == 'IfStatement':
            self.stmt_if(s)
        elif k == 'ReturnStatement':
            arg = self.ev(s.get('argument')) if s.get('argument') is not None else ident('undefined')
            self.stmts.append(stmt(intr('$return', arg)))
            self.done = True
        elif k == 'ThrowStatement':
            self.stmts.append(stmt(intr('$throw', self.ev(s.get('argument')))))
            self.done = True
        elif k == 'BreakStatement':
            self.done = True
        elif k != 'EmptyStatement':
            self.note(f'unmodelled statement {k}')

    def assign_local(self, target, value):
        b = self.it.binding(target)
        if b is None or not self.it.is_local(b):
            self.note(f'assignment to non-local {_name(target)}')
            return
        self.env[b] = value

    def expression_statement(self, e):
        if e.kind == 'AssignmentExpression':
            left, right, op = e.get('left'), e.get('right'), e.attr('operator', '=')
            if left.kind == 'Identifier':
                value = self.ev(right)
                if op != '=':
                    b = self.it.binding(left)
                    value = make('BinaryExpression', operator=op[:-1],
                                 left=clone(self.env.get(b, ident('undefined'))), right=value)
                self.assign_local(left, value)
                return
            if left.kind == 'MemberExpression':
                self.store(left, right)
                return
        if e.kind == 'SequenceExpression':
            for part in e.children_in('expressions'):
                self.expression_statement(part)
            return
        self.ev(e, discard=True)

    def store(self, left, right):
        # scope.length = n
        if not left.attr('computed', False) and left.get('property').attr('name') == 'length' \
                and left.get('object').kind == 'Identifier':
            b = self.it.binding(left.get('object'))
            if b is not None and not self.it.is_local(b):
                self.tag(b, 'length_write')
                value = self.ev(right)
                if self.roles.get(b) == SCOPE:
                    self.stmts.append(stmt(intr('$scope_len', value)))
                else:
                    self.stmts.append(stmt(make('AssignmentExpression', operator='=',
                                                left=make('MemberExpression', object=ident(b.name),
                                                          property=ident('length'), computed=False),
                                                right=value)))
                return
        found = self.chain(left)
        if found is not None:
            b, idx_nodes = found
            indices = [self.ev(i) for i in idx_nodes]
            self.access(b, indices, value=self.ev(right), write=True)
            return
        obj = self.ev(left.get('object'))
        if left.attr('computed', False):
            prop = self.ev(left.get('property'))
            target = make('MemberExpression', object=obj, property=prop, computed=True)
        else:
            target = make('MemberExpression', object=obj,
                          property=ident(left.get('property').attr('name')), computed=False)
        self.stmts.append(stmt(make('AssignmentExpression', operator='=', left=target, right=self.ev(right))))

    def stmt_if(self, s):
        test = self.ev(s.get('test'))
        then_body = self.branch(s.get('consequent'))
        else_body = self.branch(s.get('alternate')) if s.get('alternate') is not None else None
        for body in (then_body, else_body or []):
            for inner in body:
                call = inner.get('expression') if inner.kind == 'ExpressionStatement' else None
                if call is None or call.kind != 'CallExpression' \
                        or _name(call.get('callee')) not in ('$jump', '$setkey'):
                    self.note('effects other than jump/setkey inside a conditional')
                    break
        self.stmts.append(make('IfStatement', test=test,
                               consequent=make('BlockStatement', body=then_body),
                               alternate=make('BlockStatement', body=else_body) if else_body else None))

    def branch(self, s):
        saved = (self.stmts, dict(self.env), self.consumed, len(self.out))
        self.stmts = []
        done_before = self.done
        self.run([s])
        body = self.stmts
        if (self.consumed, len(self.out)) != saved[2:]:
            self.note('stack effect inside a conditional')
        self.stmts, self.env = saved[0], saved[1]
        self.done = done_before
        return body


def _summarise(interp, roles, facts, opcode, case):
    summary = Summary(opcode)
    ev = HandlerEval(interp, roles, facts, summary)
    try:
        ev.run(case.children_in('consequent'))
    except Exception as exc:                       # keep going; report it
        summary.notes.append(f'summariser error: {type(exc).__name__}: {exc}')
    summary.body = ev.stmts
    summary.pops = ev.consumed
    summary.depth = max(ev.consumed, ev.max_depth)
    summary.pushes = ev.out
    summary.explicit = ev.explicit
    return summary


def infer_roles(interp, facts):
    roles = {interp.stack: STACK}
    for b, tags in facts.tags.items():
        if b in roles:
            continue
        if tags['idx2'] or tags['length_write']:
            roles[b] = SCOPE
        elif tags['ext_get']:
            roles[b] = EXTERNALS
        elif tags['idx1_read'] and tags['idx1_write']:
            roles[b] = LOCALS
        elif tags['idx1_read']:
            roles[b] = ARGS if interp.is_param(b) else CONSTS
    consts = {b for b, r in roles.items() if r == CONSTS}
    for b, sources in facts.ld_from.items():
        if b not in roles and any(s in consts for s in sources):
            roles[b] = GLOBALS
    for b, tags in facts.tags.items():
        if b in roles:
            continue
        if tags['spread_child']:
            roles[b] = SCOPE
        elif tags['spread_array'] and interp.is_param(b):
            roles[b] = ARGS
    return roles


class Analysis:
    def __init__(self, interp, roles, summaries):
        self.interp = interp
        self.roles = roles
        self.summaries = summaries

    def report(self):
        it = self.interp
        lines = ['interpreter: %s   (factory %s, fetch %s)' % (
            _name(it.func.get('id')) or '<anonymous>', it.outer_name, it.fetch_name)]
        lines.append('bytecode=%s pc=%s key=%s' % (it.bytes_name, it.pc_name, it.key_name))
        lines.append('helpers:  ' + ', '.join(repr(h) for h in it.helpers.values()))
        lines.append('roles:    ' + ', '.join(
            f'{b.name}={r}' for b, r in sorted(self.roles.items(), key=lambda kv: kv[1])))
        clean = sum(1 for s in self.summaries.values() if not s.notes)
        lines.append(f'handlers: {len(self.summaries)}   fully modelled: {clean}')
        for s in self.summaries.values():
            for note in s.notes:
                lines.append(f'  op {s.opcode}: {note}')
        return '\n'.join(lines)


def analyse(session):
    interp = Interpreter(session)
    facts = Facts()
    for op, case in interp.cases.items():                 # discovery pass
        _summarise(interp, {interp.stack: STACK}, facts, op, case)
    roles = infer_roles(interp, facts)
    summaries = {op: _summarise(interp, roles, None, op, case) for op, case in interp.cases.items()}
    return Analysis(interp, roles, dict(sorted(summaries.items())))


# ---------------------------------------------------------------------------
# Cross-build comparison
# ---------------------------------------------------------------------------

def match_builds(a, b):
    """Pair up opcodes of two builds whose summaries are semantically identical.

    Returns (pairs, ambiguous, only_a, only_b); pairs are (op_a, op_b) for
    summaries that are unique on both sides."""
    index_a, index_b = defaultdict(list), defaultdict(list)
    for op, s in a.summaries.items():
        index_a[s.fingerprint()].append(op)
    for op, s in b.summaries.items():
        index_b[s.fingerprint()].append(op)
    pairs, ambiguous = [], []
    for fp, ops_a in index_a.items():
        ops_b = index_b.get(fp)
        if not ops_b:
            continue
        if len(ops_a) == 1 and len(ops_b) == 1:
            pairs.append((ops_a[0], ops_b[0]))
        else:
            ambiguous.append((ops_a, ops_b))
    matched_a = {x for p in pairs for x in p[:1]} | {x for o, _ in ambiguous for x in o}
    matched_b = {x for p in pairs for x in p[1:]} | {x for _, o in ambiguous for x in o}
    only_a = sorted(set(a.summaries) - matched_a)
    only_b = sorted(set(b.summaries) - matched_b)
    return sorted(pairs), ambiguous, only_a, only_b


def load(path):
    source = open(path, encoding='utf-8').read()
    try:
        return Session.parse(source)
    except ImportError:
        import esprima
        return Session.from_ast(esprima.parseScript(source))


def main(argv):
    if len(argv) < 2:
        print('usage: vmsummary.py SCRIPT.js [OP ...] [--vs OTHER.js]')
        return 1
    path, rest = argv[1], argv[2:]
    other = None
    if '--vs' in rest:
        i = rest.index('--vs')
        other, rest = rest[i + 1], rest[:i]
    analysis = analyse(load(path))
    print(analysis.report())
    wanted = [int(x) for x in rest] or list(analysis.summaries)
    print()
    for op in wanted:
        print(analysis.summaries[op].render())
        print()
    if other:
        second = analyse(load(other))
        pairs, ambiguous, only_a, only_b = match_builds(analysis, second)
        print(f'matched {len(pairs)} unique, {len(ambiguous)} ambiguous groups; '
              f'unmatched: {len(only_a)} here, {len(only_b)} there')
        for x, y in pairs:
            la, lb = analysis.summaries[x].layout_kinds, second.summaries[y].layout_kinds
            print(f'  {x:>3} -> {y:<3} layout {la or "-"} -> {lb or "-"}' + ('' if la == lb else '   (encoding changed)'))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
