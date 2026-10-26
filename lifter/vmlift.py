"""
vmlift -- lift disassembled VM functions to jsir.

For every basic block the handler templates (vmsummary) are instantiated against
a symbolic operand stack:

  * `$in<k>` become the stack entries, operands become literals, `$const(i)`
    becomes the constant itself, `$ext_get(i)` is resolved through the externals
    table at the factory's call site, locals/scope/globals/args become explicit
    JS (`vm_locals[3]`, `vm_scope[d][i]`, `vm_global[name]`, `vm_args[i]`);
  * every impure operation is bound to a numbered temporary `v<n>` in
    evaluation order, and pushed values are reduced to atoms, so the stack only
    ever holds identifiers and literals;
  * values still on the stack when a block ends are spilled to numbered slots
    `s<k>` (a parallel copy, sequentialised with a temporary on cycles), and the
    next block starts with `s0..s<h-1>` as its stack. Entry heights come from a
    dataflow pass and must agree at joins;
  * control flow comes from the disassembler's edges: `$jump` / `$setkey` have no
    runtime form (the key is part of the static state), conditionals become
    branch terminators, `$return` / `$throw` become terminators.

The output is dispatcher form -- `for (;;) switch (pc) { case n: ... }` -- which
is always valid JavaScript and is the shape state-machine unflatteners expect;
a structurer can replace it later without touching anything above this layer.
Child VMs become lifted factories `vm_<path>(scope) -> function`.

Modes
    production   resolved externals, children as lifted functions, self-contained
    test         faithful VM-state parameters and a factory hook, so the lifted
                 function can be run against the original in vmlcheck

Usage:  python vmlift.py SCRIPT.js [--test] [--structure] [--simplify] [--function PATH]
        --structure restores if/else and loops; --simplify also folds temporaries back into
        expressions (implies --structure; production mode only)
"""

import json
import re
import sys
from collections import defaultdict

from jsir import node as make, clone, walk, values as V
from jsir.schema import FUNCTION_KINDS
from lifter.vmsummary import _name, EXTERNALS, GLOBALS, CONSTS
from lifter import vmloader as L
from lifter import vmdisasm as D

OPERAND_RE = re.compile(r'^\$([bvw])(\d+)$')
IDENT_RE = re.compile(r'^[A-Za-z_$][A-Za-z0-9_$]*$')


# ---------------------------------------------------------------------------
# jsir node helpers
# ---------------------------------------------------------------------------

def ident(name):
    return make('Identifier', name=name)


def lit(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value < 0:
        return make('UnaryExpression', operator='-', prefix=True, argument=make('Literal', value=-value))
    return make('Literal', value=value)


def num(value):
    return lit(float(value))


def member(obj, key, computed=True):
    return make('MemberExpression', object=obj, property=key, computed=computed)


def call(callee, args):
    return make('CallExpression', callee=callee, arguments=list(args))


def assign(left, right):
    return make('AssignmentExpression', operator='=', left=left, right=right)


def xs(expression):
    return make('ExpressionStatement', expression=expression)


def const_node(value):
    return ident('undefined') if value is V.UNDEF else lit(value)


# ---------------------------------------------------------------------------
# A small JS printer (dependency-free; the pipeline's own escodegen can be used instead)
# ---------------------------------------------------------------------------

def js_expr(n):
    k = n.kind
    if k == 'Identifier':
        return n.attr('name')
    if k == 'ThisExpression':
        return 'this'
    if k == 'Literal':
        v = n.attr('value')
        if v is V.NULL:
            return 'null'
        if isinstance(v, bool):
            return 'true' if v else 'false'
        if isinstance(v, float):
            text = V.number_to_string(v)
            return f'({text})' if text.startswith('-') else text
        return json.dumps(v)
    if k == 'UnaryExpression':
        op = n.attr('operator')
        return f'({op}{" " if op.isalpha() else ""}{js_expr(n.get("argument"))})'
    if k == 'UpdateExpression':
        op = n.attr('operator')
        arg = js_expr(n.get('argument'))
        return f'({op}{arg})' if n.attr('prefix', True) else f'({arg}{op})'
    if k in ('BinaryExpression', 'LogicalExpression'):
        return f'({js_expr(n.get("left"))} {n.attr("operator")} {js_expr(n.get("right"))})'
    if k == 'ConditionalExpression':
        return f'({js_expr(n.get("test"))} ? {js_expr(n.get("consequent"))} : {js_expr(n.get("alternate"))})'
    if k == 'MemberExpression':
        obj = js_expr(n.get('object'))
        if n.get('object').kind in ('Literal', 'NewExpression', 'FunctionExpression'):
            obj = f'({obj})'
        if n.attr('computed', False):
            return f'{obj}[{js_expr(n.get("property"))}]'
        return f'{obj}.{n.get("property").attr("name")}'
    if k == 'CallExpression':
        callee = js_expr(n.get('callee'))
        if n.get('callee').kind in ('FunctionExpression', 'ArrowFunctionExpression', 'NewExpression'):
            callee = f'({callee})'
        return f'{callee}({", ".join(js_expr(a) for a in n.children_in("arguments"))})'
    if k == 'NewExpression':
        return f'new ({js_expr(n.get("callee"))})({", ".join(js_expr(a) for a in n.children_in("arguments"))})'
    if k == 'ArrayExpression':
        return '[' + ', '.join(js_expr(e) for e in n.children_in('elements')) + ']'
    if k == 'SpreadElement':
        return '...' + js_expr(n.get('argument'))
    if k == 'RestElement':
        return '...' + js_expr(n.get('argument'))
    if k == 'ObjectExpression':
        return '{}'
    if k == 'AssignmentExpression':
        return f'{js_expr(n.get("left"))} {n.attr("operator", "=")} {js_expr(n.get("right"))}'
    if k == 'ArrowFunctionExpression':
        params = ', '.join(js_expr(p) for p in n.children_in('params'))
        return f'({params}) => {js_expr(n.get("body"))}'
    if k == 'FunctionExpression':
        return js_function(n, '')
    raise ValueError(f'cannot print expression {k}')


def js_function(n, pad):
    name = n.get('id').attr('name') if n.get('id') is not None else ''
    params = ', '.join(js_expr(p) for p in n.children_in('params'))
    head = f'function {name}({params}) '
    return head + js_block(n.get('body'), pad)


def js_block(n, pad):
    lines = ['{']
    for s in n.children_in('body'):
        lines += js_stmt(s, pad + '  ')
    lines.append(pad + '}')
    return '\n'.join(lines)


def js_stmt(n, pad=''):
    k = n.kind
    if k == 'ExpressionStatement':
        return [pad + js_expr(n.get('expression')) + ';']
    if k == 'VariableDeclaration':
        decls = []
        for d in n.children_in('declarations'):
            name = d.get('id').attr('name')
            decls.append(name if d.get('init') is None else f'{name} = {js_expr(d.get("init"))}')
        return [f'{pad}{n.attr("kind", "var")} {", ".join(decls)};']
    if k == 'ReturnStatement':
        return [pad + 'return' + (' ' + js_expr(n.get('argument')) if n.get('argument') is not None else '') + ';']
    if k == 'ThrowStatement':
        return [f'{pad}throw {js_expr(n.get("argument"))};']
    if k in ('ContinueStatement', 'BreakStatement'):
        word = 'continue' if k == 'ContinueStatement' else 'break'
        label = n.get('label')
        return [pad + word + (' ' + label.attr('name') if label is not None else '') + ';']
    if k == 'LabeledStatement':
        inner = js_stmt(n.get('body'), pad)
        inner[0] = f'{pad}{n.get("label").attr("name")}: ' + inner[0][len(pad):]
        return inner
    if k == 'WhileStatement':
        return [f'{pad}while ({js_expr(n.get("test"))}) ' + js_block(n.get('body'), pad)]
    if k == 'DoWhileStatement':
        return [f'{pad}do ' + js_block(n.get('body'), pad) + f' while ({js_expr(n.get("test"))});']
    if k == 'EmptyStatement':
        return [pad + ';']
    if k == 'BlockStatement':
        return [pad + js_block(n, pad)]
    if k == 'IfStatement':
        lines = [f'{pad}if ({js_expr(n.get("test"))}) ' + js_block(n.get('consequent'), pad)]
        alt = n.get('alternate')
        if alt is not None and alt.kind == 'IfStatement':           # else if
            tail = js_stmt(alt, pad)
            lines[-1] += ' else ' + tail[0][len(pad):]
            lines += tail[1:]
        elif alt is not None:
            lines[-1] += ' else ' + js_block(alt, pad)
        return lines
    if k == 'ForStatement':
        return [f'{pad}for (;;) ' + js_block(n.get('body'), pad)]
    if k == 'SwitchStatement':
        lines = [f'{pad}switch ({js_expr(n.get("discriminant"))}) {{']
        for c in n.children_in('cases'):
            comment = c.attr('comment')
            label = 'default' if c.get('test') is None else f'case {js_expr(c.get("test"))}'
            lines.append(f'{pad}  {label}:' + (f'  // {comment}' if comment else ''))
            for s in c.children_in('consequent'):
                lines += js_stmt(s, pad + '    ')
        lines.append(pad + '}')
        return lines
    if k == 'TryStatement':
        h = n.get('handler')
        return [f'{pad}try ' + js_block(n.get('block'), pad) +
                f' catch ({js_expr(h.get("param"))}) ' + js_block(h.get('body'), pad)]
    if k == 'FunctionDeclaration':
        return [pad + js_function(n, pad)]
    if k == 'Program':
        out = []
        for s in n.children_in('body'):
            out += js_stmt(s, pad)
            out.append('')
        return out
    raise ValueError(f'cannot print statement {k}')


def to_js(n):
    return '\n'.join(js_stmt(n) if n.is_statement or n.kind == 'Program' else [js_expr(n)])


# ---------------------------------------------------------------------------
# What the lifter needs to know about the build
# ---------------------------------------------------------------------------

class LiftContext:
    def __init__(self, session, layout, analysis, factory_call, mode='production'):
        self.session, self.layout, self.analysis = session, layout, analysis
        self.interp = analysis.interp
        self.mode = mode
        self.externals = self._externals(factory_call)
        self.free_map = self._free_map()
        self.global_source = self._global_source()
        self.extras = sorted({n for s in analysis.summaries.values() for n in s.free
                              if n in self.free_map and self.free_map[n].startswith('vm_p')})

    def _role_binding(self, role):
        return next((b for b, r in self.analysis.roles.items() if r == role), None)

    def _externals(self, factory_call):
        b = self._role_binding(EXTERNALS)
        out = {}
        if b is None or factory_call is None:
            return out
        params = [_name(p) for p in self.interp.outer.children_in('params')]
        args = factory_call.children_in('arguments')
        if b.name not in params or params.index(b.name) >= len(args):
            return out
        table = args[params.index(b.name)]
        if table.kind != 'ObjectExpression':
            return out
        for p in table.children_in('properties'):
            key = p.get('key')
            k = int(key.attr('value')) if key.kind == 'Literal' and isinstance(key.attr('value'), float) \
                else (key.attr('value') if key.kind == 'Literal' else _name(key))
            pair = p.get('value').children_in('elements') if p.get('value').kind == 'ArrayExpression' else []
            if pair and pair[0].kind == 'ArrowFunctionExpression' and pair[0].get('body').kind != 'BlockStatement':
                out[k] = pair[0].get('body')
        return out

    def _free_map(self):
            """Interpreter parameters / outer variables that handlers reference, by what they are."""
            m = {}
            call_node = getattr(self.layout, 'interp_call_node', None)  # <--- Read the AST node
            params = self.interp.func.children_in('params')
            if call_node is not None:
                for i, (p, a) in enumerate(zip(params, call_node.children_in('arguments'))):
                    if a.kind == 'ArrowFunctionExpression' and a.get('body').kind == 'ThisExpression':
                        m[p.attr('name')] = 'vm_this_fn'
                    elif a.kind == 'MetaProperty':
                        m[p.attr('name')] = 'undefined'
                    else:
                        m[p.attr('name')] = f'vm_p{i}'
            for a in walk(self.interp.outer, kinds={'AssignmentExpression'}):
                right = a.get('right')
                if right.kind in FUNCTION_KINDS and call_node is not None \
                        and any(n is call_node for n in walk(right)) and _name(a.get('left')):
                    m[_name(a.get('left'))] = 'vm_self'
            return m

    def _global_source(self):
        b = self._role_binding(GLOBALS)
        if b is not None:
            for v in L._written_values(b):
                if v is not None:
                    return v
        return None


# ---------------------------------------------------------------------------
# Lifting one function
# ---------------------------------------------------------------------------

class Term:
    def __init__(self, kind, **kw):
        self.kind = kind
        self.__dict__.update(kw)

    def successors(self):
        if self.kind == 'goto':
            return [self.target]
        if self.kind == 'branch':
            return [self.t, self.f]
        return []


class LBlock:
    def __init__(self, bid, state):
        self.id, self.state = bid, state
        self.stmts, self.term = [], None
        self.entry_h = self.exit_h = 0
        self.exc_targets = set()


class Lifted:
    def __init__(self, name):
        self.name = name
        self.blocks = []           # ordered; blocks[0] is the entry
        self.notes = []
        self.max_stack = 0
        self.temps = 0
        self.decoded = None
        self.uses = set()          # which VM-state names the body references
        self.has_exc = False


class _Inst:
    """Per-instruction instantiation state."""

    def __init__(self, ops, ins):
        self.ops, self.ins, self.tmap = ops, ins, {}


class FunctionLifter:
    def __init__(self, ctx, decoded, dis, analysis, name, child_names):
        self.ctx, self.decoded, self.dis, self.analysis = ctx, decoded, dis, analysis
        self.name, self.child_names = name, child_names
        self.out = Lifted(name)
        self.out.decoded = decoded
        self.ntemp = 0
        self.test = ctx.mode == 'test'

    # -- helpers ------------------------------------------------------------

    def note(self, text):
        self.out.notes.append(text)

    def fresh(self):
        name = f'v{self.ntemp}'
        self.ntemp += 1
        return name

    def use(self, name):
        self.out.uses.add(name)
        return ident(name)

    def atom(self, value, stmts):
        if value.kind in ('Identifier', 'Literal'):
            return value
        name = self.fresh()
        stmts.append(xs(assign(ident(name), value)))
        return ident(name)

    # -- expression lowering -------------------------------------------------

    def expr(self, n, c):
        k = n.kind
        if k == 'Identifier':
            return self.lower_ident(n, c)
        if k == 'Literal':
            return clone(n)
        if k == 'UnaryExpression':
            return make(k, operator=n.attr('operator'), prefix=True, argument=self.expr(n.get('argument'), c))
        if k in ('BinaryExpression', 'LogicalExpression'):
            return make(k, operator=n.attr('operator'), left=self.expr(n.get('left'), c), right=self.expr(n.get('right'), c))
        if k == 'ConditionalExpression':
            return make(k, test=self.expr(n.get('test'), c), consequent=self.expr(n.get('consequent'), c),
                        alternate=self.expr(n.get('alternate'), c))
        if k == 'MemberExpression':
            obj = self.expr(n.get('object'), c)
            if n.attr('computed', False):
                return member(obj, self.expr(n.get('property'), c))
            return member(obj, ident(n.get('property').attr('name')), computed=False)
        if k == 'CallExpression':
            return self.lower_call(n, c)
        if k == 'NewExpression':
            return make('NewExpression', callee=self.expr(n.get('callee'), c), arguments=self.args(n, c))
        if k == 'ArrayExpression':
            return make('ArrayExpression', elements=self.args(n, c, 'elements'))
        if k == 'SpreadElement':
            return make('SpreadElement', argument=self.expr(n.get('argument'), c))
        if k == 'ObjectExpression':
            return make('ObjectExpression', properties=[])
        if k == 'AssignmentExpression':
            return assign(self.expr(n.get('left'), c), self.expr(n.get('right'), c))
        raise ValueError(f'cannot lower {k}')

    def args(self, n, c, slot='arguments'):
        out = []
        for a in n.children_in(slot):
            low = self.expr(a, c)
            if low.kind == 'SpreadElement' and low.get('argument').kind == 'ArrayExpression' \
                    and all(e.kind in ('Identifier', 'Literal') for e in low.get('argument').children_in('elements')):
                out += [clone(e) for e in low.get('argument').children_in('elements')]     # ...[a, b] -> a, b
            else:
                out.append(low)
        return out

    def lower_ident(self, n, c):
        name = n.attr('name')
        if name.startswith('$in') and name[3:].isdigit():
            return clone(c.ins[int(name[3:])])
        m = OPERAND_RE.match(name)
        if m:
            return num(c.ops[name[1:]])
        if name in c.tmap:
            return clone(c.tmap[name])
        if name == '$externals':
            return self.use('vm_ext')
        if name == '$new_target':
            return ident('undefined')
        mapped = self.ctx.free_map.get(name)
        if mapped is not None:
            if mapped == 'vm_self':
                return ident(self.name)
            if mapped == 'undefined':
                return ident('undefined')
            return self.use(mapped)
        return ident(name)

    def lower_call(self, n, c):
        name = _name(n.get('callee'))
        if name is None or not name.startswith('$'):
            return call(self.expr(n.get('callee'), c), self.args(n, c))
        a = self.args(n, c)
        if name == '$const':
            idx = a[0].attr('value') if a and a[0].kind == 'Literal' else None
            consts = self.decoded.constants
            if idx is None or not 0 <= int(idx) < len(consts):
                self.note(f'$const with unresolved index {idx}')
                return ident('undefined')
            return const_node(consts[int(idx)])
        if name == '$arg':
            return member(self.use('vm_args'), a[0])
        if name == '$local':
            return member(self.use('vm_locals'), a[0])
        if name == '$scope':
            return member(member(self.use('vm_scope'), a[0]), a[1])
        if name == '$scope_row':
            return member(self.use('vm_scope'), a[0])
        if name == '$global':
            return member(self.use('vm_global'), a[0])
        if name == '$scope_snapshot':
            return make('ArrayExpression', elements=[make('SpreadElement', argument=self.use('vm_scope'))])
        if name == '$args_copy':
            return make('ArrayExpression', elements=[make('SpreadElement', argument=self.use('vm_args'))])
        if name == '$call':
            return call(member(a[0], ident('call'), computed=False), a[1:])
        if name == '$ext_get':
            idx = a[0].attr('value') if a and a[0].kind == 'Literal' else None
            if not self.test:
                resolved = self.ctx.externals.get(int(idx)) if idx is not None else None
                if resolved is None:
                    self.note(f'external {idx} not resolvable from the call site')
                    return ident('undefined')
                return clone(resolved)
            return call(member(member(self.use('vm_ext'), a[0]), num(0)), [])
        if name == '$child_vm':
            if self.test:
                return call(self.use('vm_factory'), a)
            target = self.child_names.get(self.current_child)
            if target is None:
                self.note('child VM instantiated but its function was not decoded')
                return ident('undefined')
            return call(ident(target), [a[1]] if len(a) > 1 else [])
        raise ValueError(f'unknown intrinsic {name}')

    # -- statements ---------------------------------------------------------------

    def exec_instr(self, instr, st, stmts):
        summ = instr.summary
        ops = {k: float(v) for k, v in instr.operands.items()}
        if len(st) < summ.depth:
            self.note(f'{instr.pc:#06x}: stack underflow (needs {summ.depth}, has {len(st)})')
            st[:0] = [ident('undefined') for _ in range(summ.depth - len(st))]
        c = _Inst(ops, [st[-1 - k] for k in range(summ.depth)])
        c.stack = st
        if summ.pops:
            del st[len(st) - summ.pops:]
        self.current_child = instr.child
        term = None
        for s in summ.body:
            term = self.stmt(s, c, stmts, instr) or term
        for p in summ.pushes:
            st.append(self.atom(self.expr(p, c), stmts))
        return term

    def stmt(self, s, c, stmts, instr):
        k = s.kind
        if k == 'VariableDeclaration':
            d = s.children_in('declarations')[0]
            self.let(d.get('id').attr('name'), d.get('init'), c, stmts)
            return None
        if k == 'IfStatement':
            return self.branch(s, c, stmts, instr)
        if k != 'ExpressionStatement':
            self.note(f'{instr.pc:#06x}: unmodelled statement {k}')
            return None
        e = s.get('expression')
        name = _name(e.get('callee')) if e.kind == 'CallExpression' else None
        if name is None or not name.startswith('$'):
            stmts.append(xs(self.expr(e, c)))
            return None
        a = self.args(e, c)
        if name == '$push':
            c.stack.append(self.atom(a[0], stmts))
        elif name == '$return':
            return Term('return', value=self.atom(a[0], stmts))
        elif name == '$throw':
            return Term('throw', value=self.atom(a[0], stmts))
        elif name in ('$jump', '$setkey'):
            return None                                   # the key and pc are static state, not runtime
        elif name == '$poke':
            self.note(f'{instr.pc:#06x}: SELF-MODIFYING write ignored')
        elif name == '$set_local':
            stmts.append(xs(assign(member(self.use('vm_locals'), a[0]), a[1])))
        elif name == '$set_scope':
            stmts.append(xs(assign(member(member(self.use('vm_scope'), a[0]), a[1]), a[2])))
        elif name == '$set_scope_row':
            stmts.append(xs(assign(member(self.use('vm_scope'), a[0]), a[1])))
        elif name == '$scope_len':
            stmts.append(xs(assign(member(self.use('vm_scope'), ident('length'), computed=False), a[0])))
        elif name == '$set_global':
            stmts.append(xs(assign(member(self.use('vm_global'), a[0]), a[1])))
        else:
            stmts.append(xs(self.lower_call(e, c)))
        return None

    def let(self, name, init, c, stmts):
        iname = _name(init.get('callee')) if init.kind == 'CallExpression' else None
        if iname == '$pop':
            c.tmap[name] = c.stack.pop() if c.stack else ident('undefined')
            return
        if iname == '$peek':
            k = int(self.expr(init.children_in('arguments')[0], c).attr('value'))
            c.tmap[name] = clone(c.stack[-1 - k])
            return
        if iname == '$popn':
            n = int(self.expr(init.children_in('arguments')[0], c).attr('value'))
            items = [c.stack.pop() for _ in range(n)]
            items.reverse()
            c.tmap[name] = make('ArrayExpression', elements=items)
            return
        v = self.fresh()
        stmts.append(xs(assign(ident(v), self.expr(init, c))))
        c.tmap[name] = ident(v)

    def branch(self, s, c, stmts, instr):
        test = self.expr(s.get('test'), c)
        cond = test if test.kind == 'Literal' else self.atom(test, stmts)
        t = next((e for e in instr.edges if e.cond is True), None)
        f = next((e for e in instr.edges if e.cond is False), None)
        if t is None or f is None:
            self.note(f'{instr.pc:#06x}: conditional with a missing arm; treated as unconditional')
            e = t or f
            return Term('goto', target=(e.pc, e.key)) if e else None
        return Term('branch', cond=cond, t=(t.pc, t.key), f=(f.pc, f.key))

    # -- blocks ----------------------------------------------------------------------

    def spill(self, st, stmts):
        moves = [(k, v) for k, v in enumerate(st)
                 if not (v.kind == 'Identifier' and v.attr('name') == f's{k}')]
        while moves:
            need = {v.attr('name') for _, v in moves if v.kind == 'Identifier'}
            for i, (k, v) in enumerate(moves):
                if f's{k}' not in need:
                    stmts.append(xs(assign(ident(f's{k}'), clone(v))))
                    moves.pop(i)
                    break
            else:                                         # a cycle: park one destination in a temporary
                k, _ = moves[0]
                tmp = self.fresh()
                stmts.append(xs(assign(ident(tmp), ident(f's{k}'))))
                moves = [(kk, ident(tmp) if vv.kind == 'Identifier' and vv.attr('name') == f's{k}' else vv)
                         for kk, vv in moves]

    def lift_block(self, lb, instrs, height):
        st = [ident(f's{k}') for k in range(height)]
        lb.entry_h = height
        term = None
        cur = None
        for instr in instrs:
            if self.has_exc:                                  # which handler covers this instruction (-1: none)
                handler = (instr.exc[0].pc, instr.exc[0].key) if instr.exc else None
                hid = self.ids.get(handler, -1) if handler is not None else -1
                if handler is not None and handler in self.ids:
                    lb.exc_targets.add(handler)
                if hid != cur:
                    lb.stmts.append(xs(assign(ident('vm_h'), num(hid))))
                    cur = hid
            if instr.error or instr.summary is None:
                self.note(f'{instr.pc:#06x}: cannot lift an undecodable instruction')
                term = Term('throw', value=ident('undefined'))
                break
            term = self.exec_instr(instr, st, lb.stmts) or term
            self.out.max_stack = max(self.out.max_stack, len(st))
        last = instrs[-1]
        if term is None:
            if len(last.edges) == 1:
                term = Term('goto', target=(last.edges[0].pc, last.edges[0].key))
            else:
                self.note(f'{last.pc:#06x}: block ends without a successor')
                term = Term('throw', value=ident('undefined'))
        if term.kind in ('goto', 'branch'):
            self.spill(st, lb.stmts)
        lb.term, lb.exit_h = term, len(st)

    def lift(self):
        blocks = self.dis.blocks()
        entry = (0, self.decoded.key)
        order = [entry] + sorted(s for s in blocks if s != entry)
        ids = {s: i for i, s in enumerate(order)}
        self.ids = ids
        self.has_exc = self.out.has_exc = any(i.exc for i in self.dis.instructions.values())
        heights, lifted, queue = {entry: 0}, {}, [entry]
        while queue:
            state = queue.pop(0)
            if state in lifted or state not in blocks:
                continue
            lb = LBlock(ids[state], state)
            self.lift_block(lb, blocks[state], heights[state])
            lifted[state] = lb
            for succ in lb.term.successors() + sorted(lb.exc_targets):
                exc_entry = succ in lb.exc_targets and succ not in lb.term.successors()
                height_in = 1 if exc_entry else lb.exit_h              # a handler starts with [exception]
                if succ not in ids:
                    self.note(f'edge to {succ[0]:#06x}/{succ[1]:#04x} which is not a decoded block')
                elif succ in heights and heights[succ] != height_in:
                    self.note(f'stack height conflict entering {succ[0]:#06x}/{succ[1]:#04x}: '
                              f'{heights[succ]} vs {height_in}')
                elif succ not in heights:
                    heights[succ] = height_in
                    queue.append(succ)
        self.out.blocks = [lifted[s] for s in order if s in lifted]
        self.out.temps = self.ntemp
        return self.out


# ---------------------------------------------------------------------------
# Emission: dispatcher form
# ---------------------------------------------------------------------------

def _decl(names):
    return make('VariableDeclaration', kind='var',
                declarations=[make('VariableDeclarator', id=ident(n), init=None) for n in names])


def dispatcher_body(lifted, ids, guard=False):
    cases = []
    for lb in lifted.blocks:
        body = list(lb.stmts)
        t = lb.term
        if t.kind == 'goto':
            body += [xs(assign(ident('pc'), num(ids[t.target]))), make('ContinueStatement', label=None)]
        elif t.kind == 'branch':
            body += [make('IfStatement', test=clone(t.cond),
                          consequent=make('BlockStatement', body=[xs(assign(ident('pc'), num(ids[t.t])))]),
                          alternate=make('BlockStatement', body=[xs(assign(ident('pc'), num(ids[t.f])))])),
                     make('ContinueStatement', label=None)]
        elif t.kind == 'return':
            body.append(make('ReturnStatement', argument=t.value))
        else:
            body.append(make('ThrowStatement', argument=t.value))
        case = make('SwitchCase', test=num(lb.id), consequent=body)
        case.attrs['comment'] = f'{lb.state[0]:#06x} key {lb.state[1]:#04x}'
        cases.append(case)
    default = make('SwitchCase', test=None, consequent=[make('ThrowStatement', argument=call(ident('Error'), []))])
    switch = make('SwitchStatement', discriminant=ident('pc'), cases=cases + [default])
    if lifted.has_exc:
        rethrow = make('IfStatement', test=make('BinaryExpression', operator='<', left=ident('vm_h'), right=num(0)),
                       consequent=make('BlockStatement', body=[make('ThrowStatement', argument=ident('vm_e'))]), alternate=None)
        catch = make('CatchClause', param=ident('vm_e'), body=make('BlockStatement', body=[
            rethrow, xs(assign(ident('s0'), ident('vm_e'))), xs(assign(ident('pc'), ident('vm_h'))),
            xs(assign(ident('vm_h'), num(-1))), make('ContinueStatement', label=None)]))
        switch = make('TryStatement', block=make('BlockStatement', body=[switch]), handler=catch, finalizer=None)
    loop_body = [switch]
    if guard:                                   # test mode only: bound runaway loops (outside the try, so no handler eats it)
        loop_body.insert(0, make('IfStatement', test=make('BinaryExpression', operator='>', left=make('UpdateExpression', operator='++', prefix=True, argument=ident('vm_steps')), right=num(200000)),
                                 consequent=make('BlockStatement', body=[make('ThrowStatement', argument=call(ident('Error'), [lit('budget:steps')]))]), alternate=None))
    loop = make('ForStatement', init=None, test=None, update=None, body=make('BlockStatement', body=loop_body))
    names = [f'v{i}' for i in range(lifted.temps)] + [f's{i}' for i in range(max(lifted.max_stack, 1 if lifted.has_exc else 0))]
    prologue = ([_decl(names)] if names else []) + ([make('VariableDeclaration', kind='var', declarations=[
        make('VariableDeclarator', id=ident('vm_steps'), init=num(0))])] if guard else []) + [
        make('VariableDeclaration', kind='var', declarations=[
            make('VariableDeclarator', id=ident('pc'), init=num(0))] + (
            [make('VariableDeclarator', id=ident('vm_h'), init=num(-1))] if lifted.has_exc else []))]
    return prologue + [loop]


CORE_PARAMS = ['vm_this_fn', 'vm_args', 'vm_locals', 'vm_scope', 'vm_global', 'vm_ext', 'vm_factory']


def core_function(lifted, ctx, ids, name='vm_core'):
    """Test mode: every VM-state name is an explicit parameter, mirroring the interpreter's own."""
    params = CORE_PARAMS + list(sorted(n for n in lifted.uses if n.startswith('vm_p'))) + [lifted.name]
    return make('FunctionDeclaration', id=ident(name), params=[ident(p) for p in params],
                body=make('BlockStatement', body=dispatcher_body(lifted, ids, guard=True)))


def production_function(lifted, ctx, ids, factory_name, structure=False):
    """vm_<path>(vm_scope0) -> the VM function, with its own locals/scope set up like the original wrapper.

    structure=True restores if/else and loops (vmstructure); a function it cannot
    structure keeps dispatcher form and gets a note saying why."""
    prologue = [make('VariableDeclaration', kind='var', declarations=[
        make('VariableDeclarator', id=ident('vm_locals'), init=make('ArrayExpression', elements=[])),
        make('VariableDeclarator', id=ident('vm_scope'), init=make('ArrayExpression', elements=[
            make('SpreadElement', argument=ident('vm_scope0')), ident('vm_locals')]))])]
    if 'vm_this_fn' in lifted.uses:
        prologue.append(make('VariableDeclaration', kind='var', declarations=[make('VariableDeclarator', id=ident('vm_this_fn'),
            init=make('ArrowFunctionExpression', params=[], body=make('ThisExpression'), expression=True))]))
    if 'vm_global' in lifted.uses and ctx.global_source is not None:
        prologue.append(make('VariableDeclaration', kind='var', declarations=[
            make('VariableDeclarator', id=ident('vm_global'), init=clone(ctx.global_source))]))
    body = None
    if structure:
        from lifter import vmstructure
        try:
            body = vmstructure.structured_body(lifted, ids)
        except vmstructure.Unstructurable as why:
            lifted.notes.append(f'kept dispatcher form: {why}')
    if body is None:
        body = dispatcher_body(lifted, ids)
    inner = make('FunctionExpression', id=ident(lifted.name), params=[make('RestElement', argument=ident('vm_args'))],
                 body=make('BlockStatement', body=prologue + body))
    return make('FunctionDeclaration', id=ident(factory_name), params=[ident('vm_scope0')],
                body=make('BlockStatement', body=[make('ReturnStatement', argument=inner)]))


# ---------------------------------------------------------------------------
# A whole VM tree
# ---------------------------------------------------------------------------

class FnNode:
    def __init__(self, decoded, path, parent=None):
        self.decoded, self.path, self.parent = decoded, path, parent
        self.children = {}               # constant index -> FnNode
        base = decoded.name if isinstance(decoded.name, str) and IDENT_RE.match(decoded.name) else 'fn'
        self.factory = 'vm_main' if not path else f'vm_{path}'
        self.name = f'{base}_{path}' if path else (base if base != 'fn' else 'vm_main_fn')


def build_tree(layout, decoded, path='', parent=None, cache=None, depth=0):
    cache = cache if cache is not None else {}
    node = FnNode(decoded, path, parent)
    if depth < 8:
        for idx, child in L.decode_children(layout, decoded, set()).items():
            node.children[idx] = build_tree(layout, child, f'{path}_c{idx}' if path else f'c{idx}', node, cache, depth + 1)
    return node


def all_nodes(node):
    yield node
    for c in node.children.values():
        yield from all_nodes(c)


def lift_tree(session, layout, analysis, factory_call, payload, mode='production'):
    ctx = LiftContext(session, layout, analysis, factory_call, mode)
    root = build_tree(layout, L.decode_payload(layout, payload))
    results = []
    for node in all_nodes(root):
        dis, _ = D.disassemble(layout, analysis, node.decoded)
        child_names = {i: c.factory for i, c in node.children.items()}
        fl = FunctionLifter(ctx, node.decoded, dis, analysis, node.name, child_names)
        lifted = fl.lift()
        lifted.dis = dis
        results.append((node, lifted, fl.ids))
    return ctx, root, results


def program(ctx, results, structure=False):
    funcs = []
    for node, lifted, ids in reversed(results):             # children before parents
        funcs.append(production_function(lifted, ctx, ids, node.factory, structure))
    return make('Program', body=funcs, sourceType='script')


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    session = L.load(argv[1])
    layout, analysis = L.build_layout(session)
    payloads = L.find_payloads(session, analysis.interp)
    if not payloads:
        print('no call of the factory with a literal payload found')
        return 2
    call_node, payload = payloads[0]
    mode = 'test' if '--test' in argv else 'production'
    simplify = '--simplify' in argv and mode == 'production'
    structure = ('--structure' in argv or simplify) and mode == 'production'
    ctx, root, results = lift_tree(session, layout, analysis, call_node, payload, mode)
    want = argv[argv.index('--function') + 1] if '--function' in argv else None
    if mode == 'production' and want is None:
        prog = program(ctx, results, structure)
        if simplify:
            from lifter import vmexpr
            from jsir import Session, Tree
            vmexpr.simplify(Session(Tree(prog)))
        print(to_js(prog))
    for node, lifted, ids in results:
        if want is not None and node.path != want and node.factory != want:
            continue
        fn = core_function(lifted, ctx, ids) if mode == 'test' else production_function(lifted, ctx, ids, node.factory, structure)
        if mode == 'test' or want is not None:
            print(to_js(fn))
    for node, lifted, ids in results:
        for n in lifted.notes:
            print(f'// NOTE {node.factory}: {n}', file=sys.stderr)
    print(f'// {len(results)} functions lifted, {sum(len(l.blocks) for _, l, _ in results)} blocks', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
