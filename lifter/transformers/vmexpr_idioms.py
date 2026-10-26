"""
vmexpr_idioms -- lifter idioms: calls, [arguments], selects, joins, compound assignments.
"""

from jsir import clone, walk, values as V, node as make
from lifter.transformers import vmexpr_util
from lifter.transformers.vmexpr_util import (
    IDENT_RE,
    assignment_of,
    body_lists,
    ident,
    leaf_functions,
    names_read,
    names_written,
    tracked_bindings,
)
from lifter.transformers.vmexpr_liveness import _Liveness


def _single_assignment(block):
    if block is None:
        return None
    stmts = block.children_in('body') if block.kind == 'BlockStatement' else [block]
    return assignment_of(stmts[0]) if len(stmts) == 1 else None


def merge_selects(session):
    tree = session.tree
    index = session.scopes
    for fn in leaf_functions(tree.root):
        tracked = tracked_bindings(index, fn)
        if tracked is None:
            continue
        for parent, slot in body_lists(fn):
            items = list(parent.children_in(slot))
            for pos, s in enumerate(items):
                if not s.live or s.kind != 'IfStatement':
                    continue
                a = _single_assignment(s.get('consequent'))
                b = _single_assignment(s.get('alternate'))
                if a and b and a[0] == b[0] and a[0] in tracked:
                    value = make('ConditionalExpression', test=clone(s.get('test')),
                                 consequent=clone(a[1].get('right')), alternate=clone(b[1].get('right')))
                    tree.replace(s, make('ExpressionStatement', expression=make(
                        'AssignmentExpression', operator='=', left=ident(a[0]), right=value)))
                    continue
                if pos == 0 or s.get('alternate') is not None:
                    continue
                prev = items[pos - 1]
                d = assignment_of(prev) if prev.live else None
                c = _single_assignment(s.get('consequent'))
                if not d or not c or c[0] != d[0] or d[0] not in tracked:
                    continue
                if d[0] in names_read(c[1].get('right')):
                    continue
                test, op = s.get('test'), '&&'
                if test.kind == 'UnaryExpression' and test.attr('operator') == '!':
                    test, op = test.get('argument'), '||'
                left = d[1].get('right')
                if test.kind != 'Identifier':
                    continue
                if not (test.attr('name') == d[0] or
                        (left.kind == 'Identifier' and left.attr('name') == test.attr('name'))):
                    continue
                value = make('LogicalExpression', operator=op, left=clone(left), right=clone(c[1].get('right')))
                tree.replace(prev, make('ExpressionStatement', expression=make(
                    'AssignmentExpression', operator='=', left=ident(d[0]), right=value)))
                tree.remove(s)


def _same_ref(a, b):
    if a.kind != b.kind:
        return False
    if a.kind == 'Identifier':
        return a.attr('name') == b.attr('name')
    if a.kind == 'Literal':
        return a.attr('value') == b.attr('value') and type(a.attr('value')) is type(b.attr('value'))
    if a.kind == 'MemberExpression':
        return a.attr('computed', False) == b.attr('computed', False) \
            and _same_ref(a.get('object'), b.get('object')) and _same_ref(a.get('property'), b.get('property'))
    return False


def recover_calls(session, globals_as_identifiers=False):
    tree = session.tree
    index = session.scopes
    for fn in leaf_functions(tree.root):
        tracked = tracked_bindings(index, fn) or {}
        for call in [c for c in walk(fn.get('body'), kinds={'CallExpression'})]:
            if not call.live:
                continue
            callee, args = call.get('callee'), call.children_in('arguments')

            # Fold `vm_this_fn.call(...)` or `vm_this_fn()` -> `this`
            if callee.kind == 'Identifier' and callee.attr('name') == 'vm_this_fn':
                tree.replace(call, make('ThisExpression'))
                continue
            if callee.kind == 'MemberExpression' and callee.get('object').kind == 'Identifier' \
                    and callee.get('object').attr('name') == 'vm_this_fn':
                tree.replace(call, make('ThisExpression'))
                continue

            if callee.kind != 'MemberExpression' or callee.attr('computed', False) \
                    or callee.get('property').attr('name') != 'call' or not args:
                continue
            f, this = callee.get('object'), args[0]
            direct = None
            if f.kind == 'MemberExpression' and _same_ref(f.get('object'), this):
                direct = f                                         # o[k].call(o, ...)  ->  o[k](...)
            elif f.kind != 'MemberExpression' and (
                    (this.kind == 'Identifier' and this.attr('name') == 'undefined') or
                    (this.kind == 'Literal' and this.attr('value') is V.UNDEF)):
                direct = f                                         # f.call(undefined, ...)  ->  f(...)
            if direct is None:
                continue
            tree.replace(call, make('CallExpression', callee=clone(direct),
                                    arguments=[clone(a) for a in args[1:]]))
    for m in [m for fn in leaf_functions(tree.root) for m in walk(fn.get('body'), kinds={'MemberExpression'})]:
        p = m.get('property')
        if m.live and m.attr('computed', False) and p.kind == 'Literal' and isinstance(p.attr('value'), str) \
                and IDENT_RE.match(p.attr('value')):
            name = p.attr('value')
            m.attrs['computed'] = False
            tree.set_slot(m, 'property', ident(name))


_COMPOUND = ('+', '-', '*', '/', '%', '**', '<<', '>>', '>>>', '&', '|', '^')


def _simple_ref(e):
    if e.kind == 'Identifier':
        return True
    if e.kind == 'MemberExpression':
        if e.attr('computed', False) and e.get('property').kind != 'Literal':
            return False
        return _simple_ref(e.get('object'))
    return False


def compound_assignments(session):
    tree = session.tree
    for a in [n for n in walk(tree.root, kinds={'AssignmentExpression'})]:
        if not a.live or a.attr('operator', '=') != '=':
            continue
        right = a.get('right')
        if right.kind != 'BinaryExpression' or right.attr('operator') not in _COMPOUND:
            continue
        left = a.get('left')
        if _simple_ref(left) and _same_ref(left, right.get('left')):
            a.attrs['operator'] = right.attr('operator') + '='
            tree.set_slot(a, 'right', clone(right.get('right')))


def _member_name(m):
    if m.kind != 'MemberExpression':
        return None
    p = m.get('property')
    if not m.attr('computed', False):
        return p.attr('name')
    return p.attr('value') if p.kind == 'Literal' and isinstance(p.attr('value'), str) else None


def _call_of(stmt, owner, name):
    """`owner.name(...)` as an expression statement -> the CallExpression."""
    if stmt.kind != 'ExpressionStatement' or stmt.get('expression').kind != 'CallExpression':
        return None
    call = stmt.get('expression')
    c = call.get('callee')
    if c.kind == 'MemberExpression' and c.get('object').kind == 'Identifier' \
            and c.get('object').attr('name') == owner and _member_name(c) == name:
        return call
    return None


def _empty(e, kind):
    return e.kind == kind and not e.children_in('elements' if kind == 'ArrayExpression' else 'properties')


def _get_prototype_of(e):
    """`Object.getPrototypeOf(<arg>)` -> arg, else None."""
    if e.kind == 'CallExpression' and e.get('callee').kind == 'MemberExpression' \
            and e.get('callee').get('object').kind == 'Identifier' \
            and e.get('callee').get('object').attr('name') == 'Object' \
            and _member_name(e.get('callee')) == 'getPrototypeOf' and len(e.children_in('arguments')) == 1:
        return e.children_in('arguments')[0]
    return None


def recover_arguments(session):
    """Recover `[arguments]` and `arguments[k]` from rest parameters and array copies."""
    tree = session.tree
    index = session.scopes
    for fn in leaf_functions(tree.root):
        params = fn.children_in('params')
        if len(params) != 1 or params[0].kind != 'RestElement':
            continue
        argname = params[0].get('argument').attr('name')
        body = fn.get('body')
        if body is None or body.kind != 'BlockStatement':
            continue
        stmts = list(body.children_in('body'))

        # 1. Direct indexing: vm_args[k] -> arguments[k]
        for m in [m for m in walk(body, kinds={'MemberExpression'})]:
            if m.live and m.get('object').kind == 'Identifier' and m.get('object').attr('name') == argname:
                tree.replace(m.get('object'), ident('arguments'))

        # 2. Array copy idiom: v = [...vm_args]; ...; holder.push(v) -> holder = [arguments]
        copy_var = None
        copy_stmt = None
        for s in stmts:
            d = assignment_of(s)
            if d and d[1].get('right').kind == 'ArrayExpression':
                els = d[1].get('right').children_in('elements')
                if len(els) == 1 and els[0].kind == 'SpreadElement' \
                        and els[0].get('argument').kind == 'Identifier' and els[0].get('argument').attr('name') == argname:
                    copy_var = d[0]
                    copy_stmt = s
                    break

        if copy_var is not None:
            push_stmt, holder = None, None
            for s in stmts:
                if s.kind == 'ExpressionStatement' and s.get('expression').kind == 'CallExpression':
                    c = s.get('expression')
                    callee = c.get('callee')
                    args = c.children_in('arguments')
                    if callee.kind == 'MemberExpression' and _member_name(callee) == 'push' and len(args) == 1 \
                            and args[0].kind == 'Identifier' and args[0].attr('name') == copy_var:
                        push_stmt = s
                        holder = callee.get('object').attr('name')
                        break

            if holder is not None and push_stmt is not None:
                hdef = next((s for s in stmts if (assignment_of(s) or (None,))[0] == holder), None)
                if hdef is not None:
                    tree.replace(assignment_of(hdef)[1].get('right'), make('ArrayExpression', elements=[ident('arguments')]))

                    # Identify the bounds of the arguments-setup block
                    i_start = stmts.index(copy_stmt)
                    i_end = stmts.index(push_stmt)

                    # Gather any variables read after the push statement across the entire function
                    later_reads = set()
                    for s in body.children_in('body'):
                        if s not in stmts[i_start:i_end + 1]:
                            later_reads |= names_read(s)

                    # Remove all intermediate prototype/iterator/setup statements
                    for s in list(stmts[i_start:i_end + 1]):
                        if s is hdef:
                            continue
                        written = names_written(s)
                        if not (written & later_reads):
                            tree.remove(s)

        # 3. Drop rest parameter once nothing reads it
        scope = index.scope_of(fn)
        if scope and argname in scope.bindings and not scope.bindings[argname].reads:
            tree.set_slot(fn, 'params', [])


def _if_leaves(stmt):
    if stmt.kind != 'IfStatement' or stmt.get('alternate') is None:
        return None
    out = []
    for arm in (stmt.get('consequent'), stmt.get('alternate')):
        if arm.kind == 'IfStatement':
            sub = _if_leaves(arm)
            if sub is None:
                return None
            out += sub
        elif arm.kind == 'BlockStatement' and arm.children_in('body'):
            last = arm.children_in('body')[-1]
            if last.kind == 'IfStatement':
                sub = _if_leaves(last)
                if sub is None:
                    return None
                out += sub
            else:
                out.append(last)
        else:
            return None
    return out


def sink_joins(session):
    tree = session.tree
    index = session.scopes
    for fn in leaf_functions(tree.root):
        tracked = tracked_bindings(index, fn)
        if not tracked:
            continue
        live = _Liveness(tracked)
        live.block(fn.get('body').children_in('body'), set())
        if live.unsafe:
            continue
        for parent, slot in body_lists(fn):
            if not parent.live:
                continue
            items = list(parent.children_in(slot))
            for pos in range(len(items) - 1):
                first, use = items[pos], items[pos + 1]
                if not (first.live and use.live and first.kind == 'IfStatement'
                        and use.kind == 'ExpressionStatement') or sum(1 for _ in walk(use)) > 24:
                    continue
                leaves = _if_leaves(first)
                if not leaves or len(leaves) > 4:
                    continue
                targets = {(assignment_of(x) or (None,))[0] for x in leaves}
                if len(targets) != 1 or None in targets:
                    continue
                x = targets.pop()
                if x not in tracked or x not in names_read(use) or x in names_written(use) \
                        or x in live.after.get(use.nid, tracked.keys()):
                    continue
                for last in leaves:
                    tree.insert_after(last, [clone(use)])
                tree.remove(use)
                break
