"""
vmexpr_control -- control-flow tidying: loop tails, loop conditions, else-if chains, boolean tests.

Part of vmexpr (see vmexpr.py for the pipeline and the soundness notes).
"""

import re
import sys

from jsir import Session, Tree, node as make, clone, walk, values as V
from jsir.schema import FUNCTION_KINDS, TARGET, KEY, LABEL
from lifter.vmstructure import _negate_expr
from lifter.transformers.vmexpr_util import (
    body_lists,
    leaf_functions,
)


def _is_continue(n):
    if n.kind == 'BlockStatement':
        inner = n.children_in('body')
        return len(inner) == 1 and _is_continue(inner[0])
    return n.kind == 'ContinueStatement' and n.get('label') is None


def _tail_block(tree, block):
    """Normalise the statement list at the tail of a loop body. A `continue`
    there is a no-op, and `if (c) continue; rest` is `if (!c) { rest }`."""
    items = list(block.children_in('body'))
    if items and _is_continue(items[-1]):
        tree.remove(items[-1])
        return True
    for i, st in enumerate(items):
        if st.kind == 'IfStatement' and st.get('alternate') is None and _is_continue(st.get('consequent')):
            rest = [clone(x) for x in items[i + 1:]]
            fresh_if = make('IfStatement', test=_negate_expr(st.get('test')), consequent=make('BlockStatement', body=rest),
                            alternate=None)
            tree.set_slot(block, 'body', items[:i] + ([fresh_if] if rest else []))
            return True
    if items and items[-1].kind == 'IfStatement':
        return _tail_if(tree, items[-1])
    return False


def _tail_if(tree, node):
    changed = False
    for arm in (node.get('consequent'), node.get('alternate')):
        if arm is None:
            continue
        if arm.kind == 'BlockStatement':
            changed = _tail_block(tree, arm) or changed
        elif arm.kind == 'IfStatement':
            changed = _tail_if(tree, arm) or changed
    return changed


def loop_tail_guards(session):
    tree = session.tree
    for loop in [n for n in walk(tree.root, kinds={'WhileStatement', 'DoWhileStatement'})]:
        if loop.live and loop.get('body').kind == 'BlockStatement':
            _tail_block(tree, loop.get('body'))


def flatten_else_if(session):
    """`else { if ... }` -> `else if ...`; and inside nested blocks, an
    `if (a) { ...; return }` directly followed by a last `if (b)` becomes
    `if (a) {...} else if (b)`, since b can only run when a did not."""
    tree = session.tree
    for s in [n for n in walk(tree.root, kinds={'IfStatement'})]:
        alt = s.get('alternate') if s.live else None
        if alt is not None and alt.kind == 'BlockStatement':
            inner = alt.children_in('body')
            if len(inner) == 1 and inner[0].kind == 'IfStatement':
                tree.set_slot(s, 'alternate', clone(inner[0]))
    for fn in leaf_functions(tree.root):
        for parent, slot in body_lists(fn):
            if parent is fn.get('body') or not parent.live:
                continue                                   # top-level guard clauses read better as they are
            items = list(parent.children_in(slot))
            if len(items) < 2:
                continue
            a, b = items[-2], items[-1]
            if a.live and b.live and a.kind == 'IfStatement' and a.get('alternate') is None \
                    and b.kind == 'IfStatement' and _ends_in_jump(a.get('consequent')):
                tree.set_slot(a, 'alternate', clone(b))
                tree.remove(b)


def _ends_in_jump(block):
    stmts = block.children_in('body') if block.kind == 'BlockStatement' else [block]
    return bool(stmts) and stmts[-1].kind in ('ReturnStatement', 'ThrowStatement',
                                              'BreakStatement', 'ContinueStatement')


def tidy_conditions(session):
    """In a boolean position `!!x` is `x`, and `!(a === b)` is `a !== b`."""
    tree = session.tree
    for n in [n for n in walk(tree.root, kinds={'IfStatement', 'WhileStatement', 'DoWhileStatement',
                                                'ConditionalExpression'})]:
        if not n.live:
            continue
        test = n.get('test')
        better = _normal_test(test)
        while better is not None:
            test, changed = better, True
            better = _normal_test(test)
            if better is None:
                tree.set_slot(n, 'test', test)


def _normal_test(e):
    if e.kind == 'UnaryExpression' and e.attr('operator') == '!':
        inner = e.get('argument')
        if inner.kind == 'UnaryExpression' and inner.attr('operator') == '!':
            return clone(inner.get('argument'))
        if inner.kind == 'BinaryExpression' and inner.attr('operator') in _FLIP:
            return make('BinaryExpression', operator=_FLIP[inner.attr('operator')],
                        left=clone(inner.get('left')), right=clone(inner.get('right')))
    return None


_FLIP = {'===': '!==', '!==': '===', '==': '!=', '!=': '=='}


def loop_conditions(session):
    """while (true) { if (c) break; rest }  ->  while (!c) { rest }
    (also `if (c) break; else rest'`, which flatten_else_if may have produced)"""
    tree = session.tree
    for w in [n for n in walk(tree.root, kinds={'WhileStatement'})]:
        test = w.get('test')
        if not w.live or test.kind != 'Literal' or test.attr('value') is not True:
            continue
        body = w.get('body').children_in('body')
        if not body or body[0].kind != 'IfStatement':
            continue
        first = body[0]
        cons = first.get('consequent')
        inner = cons.children_in('body') if cons.kind == 'BlockStatement' else [cons]
        if not (len(inner) == 1 and inner[0].kind == 'BreakStatement' and inner[0].get('label') is None):
            continue
        alt = first.get('alternate')
        rest = [] if alt is None else (alt.children_in('body') if alt.kind == 'BlockStatement' else [alt])
        tree.set_slot(w, 'test', _negate_expr(first.get('test')))
        if alt is None:
            tree.remove(first)
        else:
            tree.replace_many(first, [clone(x) for x in rest])
