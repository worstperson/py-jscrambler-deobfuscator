"""
vmexpr_inline -- moving expressions: the single-use temp inliner and rematerialisation of repeated reads.

Part of vmexpr (see vmexpr.py for the pipeline and the soundness notes).
"""

import re
import sys

from jsir import Session, Tree, node as make, clone, walk, values as V
from jsir.schema import FUNCTION_KINDS, TARGET, KEY, LABEL
from lifter.vmstructure import _negate_expr
from lifter.transformers import vmexpr_util
from lifter.transformers.vmexpr_util import (
    LOOP_KINDS,
    assignment_of,
    builtin_named,
    heap_free,
    leaf_functions,
    names_read,
    names_written,
    pure,
    tracked_bindings,
)


def _events(e, cond, out):
    """Evaluation-order events of an expression: ('read', node, cond),
    ('get', node, cond) a pure read, ('eff', node, cond) anything else."""
    k = e.kind
    if k == 'Identifier':
        out.append(('read', e, cond))
    elif k in ('Literal', 'ThisExpression'):
        pass
    elif k == 'MemberExpression':
        _events(e.get('object'), cond, out)
        if e.attr('computed', False):
            _events(e.get('property'), cond, out)
        out.append(('get', e, cond))
    elif k in ('CallExpression', 'NewExpression'):
        callee = e.get('callee')
        if k == 'CallExpression' and callee.kind == 'MemberExpression':
            _events(callee.get('object'), cond, out)
            if callee.attr('computed', False):
                _events(callee.get('property'), cond, out)
            prop = callee.get('property')
            intrinsic = not callee.attr('computed', False) and prop.attr('name') in ('call', 'apply', 'bind')
            out.append(('stable' if intrinsic else 'get', callee, cond))
        else:
            _events(callee, cond, out)
        for a in e.children_in('arguments'):
            _events(a.get('argument') if a.kind == 'SpreadElement' else a, cond, out)
            if a.kind == 'SpreadElement':
                out.append(('eff', a, cond))
        out.append(('eff', e, cond))
    elif k in ('BinaryExpression',):
        _events(e.get('left'), cond, out)
        _events(e.get('right'), cond, out)
        out.append(('get', e, cond))
    elif k == 'UnaryExpression':
        _events(e.get('argument'), cond, out)
        out.append(('eff' if e.attr('operator') == 'delete' else 'get', e, cond))
    elif k == 'LogicalExpression':
        _events(e.get('left'), cond, out)
        _events(e.get('right'), True, out)
    elif k == 'ConditionalExpression':
        _events(e.get('test'), cond, out)
        _events(e.get('consequent'), True, out)
        _events(e.get('alternate'), True, out)
    elif k == 'SequenceExpression':
        for x in e.children_in('expressions'):
            _events(x, cond, out)
    elif k == 'ArrayExpression':
        for x in e.children_in('elements'):
            if x is not None:
                _events(x.get('argument') if x.kind == 'SpreadElement' else x, cond, out)
                if x.kind == 'SpreadElement':
                    out.append(('eff', x, cond))
    elif k == 'AssignmentExpression':
        left = e.get('left')
        if left.kind == 'MemberExpression':
            _events(left.get('object'), cond, out)
            if left.attr('computed', False):
                _events(left.get('property'), cond, out)
        _events(e.get('right'), cond, out)
        out.append(('eff', e, cond))
    else:
        for _s, c, _r in e.iter_children():
            _events(c, cond, out)
        out.append(('eff', e, cond))


class _Inliner:
    def __init__(self, tree, tracked, eligible):
        self.tree, self.tracked, self.eligible = tree, tracked, eligible

    # -- driver ---------------------------------------------------------------

    def block(self, parent, slot):
        pending = []
        for s in list(parent.children_in(slot)):
            if s.live:
                pending = self.step(s, pending)
        return pending

    def sub(self, n):
        if n is None:
            return
        if n.kind == 'BlockStatement':
            self.block(n, 'body')
        else:
            self.step(n, [])

    # -- timing probes ---------------------------------------------------------

    def date_ctor(self, e):
        if builtin_named(e, 'Date') and not (e.kind == 'Identifier' and e.attr('name') in self.tracked):
            return True
        if e.kind == 'Identifier' and e.attr('name') in self.tracked:
            b = self.tracked[e.attr('name')]
            if len(b.writes) == 1:
                a = b.writes[0].parent
                return a is not None and a.kind == 'AssignmentExpression' and a.get('left') is b.writes[0] \
                    and builtin_named(a.get('right'), 'Date')
        return False

    def clock(self, e):
        k = e.kind
        if k == 'Literal':
            return True
        if k == 'Identifier':
            return e.attr('name') in self.tracked
        if k == 'NewExpression':
            return self.date_ctor(e.get('callee')) and not e.children_in('arguments')
        if k in ('BinaryExpression', 'LogicalExpression'):
            return self.clock(e.get('left')) and self.clock(e.get('right'))
        if k == 'UnaryExpression':
            return e.attr('operator') != 'delete' and self.clock(e.get('argument'))
        if k == 'CallExpression' and not e.children_in('arguments') and e.get('callee').kind == 'MemberExpression':
            c = e.get('callee')
            name = c.get('property').attr('name') if not c.attr('computed', False) else None
            return (name in ('now', 'getTime', 'valueOf')) and (self.date_ctor(c.get('object')) or self.clock(c.get('object')))
        return False

    def probe(self, loop):
        """A loop whose only effect is reading the clock."""
        if not vmexpr_util.ASSUME_TIMING_LOOPS_COMMUTE or loop.kind not in ('WhileStatement', 'DoWhileStatement'):
            return False
        test = loop.get('test')
        if (test.kind == 'Literal' and test.attr('value') is True) or not self.clock(test):
            return False
        body = loop.get('body')
        for st in (body.children_in('body') if body.kind == 'BlockStatement' else [body]):
            if st.kind != 'ExpressionStatement':
                return False
            e = st.get('expression')
            d = assignment_of(st)
            if not (self.clock(e) or (d and d[0] in self.tracked and self.clock(d[1].get('right')))):
                return False
        return not any(n.kind in ('BreakStatement', 'ContinueStatement', 'ReturnStatement', 'ThrowStatement')
                       for n in walk(body))

    def do_while(self, s):
        tail = self.block(s.get('body'), 'body')
        has_continue = any(n.kind == 'ContinueStatement' for n in walk(s.get('body')))
        if tail and not has_continue:
            self.consume(s, [s.get('test')], tail)

    def step(self, s, pending):
        k = s.kind
        if pending and self.probe(s):
            names = {n for n, _ in pending}
            written = names_written(s)
            blocked = (names_read(s) & names) or any(names_read(assignment_of(ds)[1].get('right')) & written
                                                     for _n, ds in pending)
            if k == 'DoWhileStatement':
                self.do_while(s)
            else:
                self.sub(s.get('body'))
            return [] if blocked else pending
        if k == 'DoWhileStatement':
            self.do_while(s)
            return []
        roots = self.roots(s)
        if roots is not None and pending:
            pending = self.consume(s, roots, pending)
        if k == 'IfStatement':
            self.sub(s.get('consequent'))
            self.sub(s.get('alternate'))
        elif k in ('WhileStatement', 'LabeledStatement'):
            self.sub(s.get('body'))
        elif k == 'BlockStatement':
            self.block(s, 'body')
        elif k == 'TryStatement':
            self.sub(s.get('block'))
            handler = s.get('handler')
            if handler is not None:
                self.sub(handler.get('body'))
            finalizer = s.get('finalizer')
            if finalizer is not None:
                self.sub(finalizer)
        d = assignment_of(s) if k == 'ExpressionStatement' else None
        if d and d[0] in self.eligible:
            pending.append((d[0], s))
            return pending
        if d and d[0] in self.tracked and pure(d[1].get('right')) and pending:
            right, target = d[1].get('right'), d[0]
            for _n, ds in pending:
                prior = assignment_of(ds)[1].get('right')
                if target in names_read(prior) or not (pure(prior) or heap_free(right, self.tracked)):
                    return []
            return pending
        if d and d[0] in self.tracked and pure(d[1].get('right')):
            return pending
        return []

    @staticmethod
    def roots(s):
        k = s.kind
        if k == 'ExpressionStatement':
            return [s.get('expression')]
        if k in ('ReturnStatement', 'ThrowStatement'):
            return [s.get('argument')] if s.get('argument') is not None else []
        if k == 'IfStatement':
            return [s.get('test')]
        return None

    # -- the move ---------------------------------------------------------------

    def consume(self, s, roots, pending):
        events = []
        for r in roots:
            _events(r, False, events)
        names = [n for n, _ in pending]
        position, touched = {}, set()
        for idx, (kind, node, cond) in enumerate(events):
            if kind == 'read' and node.attr('name') in names:
                touched.add(node.attr('name'))
                if not cond:
                    position[node.attr('name')] = idx
        rhs = {n: assignment_of(ds)[1].get('right') for n, ds in pending}
        chosen = []
        for k in range(len(pending), 0, -1):
            group = [n for n, _ in pending[-k:]]
            if not all(n in position for n in group):
                continue
            where = [position[n] for n in group]
            if where != sorted(where) or len(set(where)) != len(where):
                continue
            if self.safe(events, set(where), {position[n]: rhs[n] for n in group}):
                chosen = group
                break
        if chosen:
            for n in chosen:
                node = events[position[n]][1]
                self.tree.replace(node, clone(rhs[n]))
            defs = {n: ds for n, ds in pending}
            for n in chosen:
                self.tree.remove(defs[n])
            pending = pending[:len(pending) - len(chosen)]
        if any(n in touched for n, _ in pending):
            return []
        return pending

    def safe(self, events, moved, exprs):
        last = max(moved)
        for idx, (kind, node, _cond) in enumerate(events[:last]):
            if idx in moved:
                continue
            if kind == 'stable':
                continue
            if kind == 'read':
                name = node.attr('name')
                if name in self.tracked or name == 'undefined':
                    continue
                kind = 'get'
            if kind == 'get':
                if all(pure(e) for q, e in exprs.items() if q > idx):
                    continue
                return False
            return False
        return True


def inline_temps(session):
    tree = session.tree
    index = session.scopes
    for fn in leaf_functions(tree.root):
        tracked = tracked_bindings(index, fn)
        if not tracked:
            continue
        eligible = set()
        for name, b in tracked.items():
            if len(b.reads) != 1 or len(b.writes) != 1:
                continue
            w = b.writes[0]
            a = w.parent
            if a is not None and a.kind == 'AssignmentExpression' and a.get('left') is w \
                    and a.attr('operator', '=') == '=' and a.parent is not None \
                    and a.parent.kind == 'ExpressionStatement' and a.parent.parent is not None \
                    and a.parent.parent.kind == 'BlockStatement':
                eligible.add(name)
        if eligible:
            _Inliner(tree, tracked, eligible).block(fn.get('body'), 'body')


def _key_of(member):
    p = member.get('property')
    if member.attr('computed', False):
        if p.kind == 'Literal':
            v = p.attr('value')
            return V.to_string(v) if not isinstance(v, bool) else None
        return None
    return p.attr('name')


def _stores_to(node, obj, key):
    """Does node (an expression or statement) write o[key], or call something on o?"""
    for n in walk(node):
        k = n.kind
        target = None
        if k == 'AssignmentExpression' or k == 'UpdateExpression':
            target = n.get('left') if k == 'AssignmentExpression' else n.get('argument')
        elif k == 'UnaryExpression' and n.attr('operator') == 'delete':
            target = n.get('argument')
        elif k in ('CallExpression', 'NewExpression'):
            c = n.get('callee')
            if c.kind == 'MemberExpression' and c.get('object').kind == 'Identifier' \
                    and c.get('object').attr('name') == obj:
                return n
            continue
        if target is not None and target.kind == 'MemberExpression' \
                and target.get('object').kind == 'Identifier' and target.get('object').attr('name') == obj:
            kk = _key_of(target)
            if kk is None or kk == key:
                return n
    return None


def _clean_path(def_stmt, read, obj, key):
    """No write to obj[key] can run between def_stmt and read."""
    node = read
    while node.parent is not None and not node.is_statement and \
            not (node.parent.kind in ('IfStatement',) and node.parent_slot == 'test') and \
            not (node.parent.kind in LOOP_KINDS and node.parent_slot == 'test'):
        node = node.parent
    root = node
    hit = _stores_to(root, obj, key)
    if hit is not None:
        ok = hit.kind == 'AssignmentExpression' and hit.get('right') is not None and \
            any(n is read for n in walk(hit.get('right'))) and _stores_to(hit.get('right'), obj, key) is None
        if not ok:
            return False
    if not root.is_statement:
        node = root.parent
    while True:
        parent = node.parent
        if parent is None:
            return False
        kind = parent.kind
        if kind == 'BlockStatement':
            items = parent.children_in('body')
            j = next(i for i, x in enumerate(items) if x is node)
            if def_stmt in items:
                i = next(i for i, x in enumerate(items) if x is def_stmt)
                return i < j and all(_stores_to(x, obj, key) is None for x in items[i + 1:j])
            if any(_stores_to(x, obj, key) is not None for x in items[:j]):
                return False
        elif kind == 'IfStatement':
            if node.parent_slot != 'test' and _stores_to(parent.get('test'), obj, key) is not None:
                return False
        elif kind in LOOP_KINDS:
            if _stores_to(parent, obj, key) is not None:
                return False
        elif kind != 'LabeledStatement' and kind != 'TryStatement':
            return False
        node = parent


def rematerialize_reads(session):
    """`state = o[3]; ... state ... state ...` -> read o[3] at each use."""
    tree = session.tree
    index = session.scopes
    for fn in leaf_functions(tree.root):
        tracked = tracked_bindings(index, fn)
        if not tracked:
            continue
        for name, b in tracked.items():
            if len(b.writes) != 1 or len(b.reads) < 1:
                continue
            w = b.writes[0]
            a = w.parent
            if not (a is not None and a.kind == 'AssignmentExpression' and a.get('left') is w
                    and a.attr('operator', '=') == '=' and a.parent is not None
                    and a.parent.kind == 'ExpressionStatement' and a.parent.parent is not None
                    and a.parent.parent.kind == 'BlockStatement'):
                continue
            rhs, stmt = a.get('right'), a.parent
            if builtin_named(rhs) and not (rhs.kind == 'Identifier' and rhs.attr('name') in tracked):
                for r in list(b.reads):
                    tree.replace(r, clone(rhs))
                continue
            if len(b.reads) < 2 or rhs.kind != 'MemberExpression' or rhs.get('object').kind != 'Identifier':
                continue
            obj = rhs.get('object').attr('name')
            key = _key_of(rhs)
            ob = tracked.get(obj)
            if key is None or ob is None or len(ob.writes) != 1:
                continue
            wp = ob.writes[0].parent
            assigned = wp.get('right') if wp.kind == 'AssignmentExpression' else \
                wp.get('init') if wp.kind == 'VariableDeclarator' else None
            if assigned is None or assigned.kind not in ('ArrayExpression', 'ObjectExpression'):
                continue
            if not all(r.parent.kind == 'MemberExpression' and r.parent.get('object') is r for r in ob.reads):
                continue
            if rhs.attr('computed', False) and _key_of(rhs) is None:
                continue
            if all(_clean_path(stmt, r, obj, key) for r in b.reads):
                for r in list(b.reads):
                    tree.replace(r, clone(rhs))
