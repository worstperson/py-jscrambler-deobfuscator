"""
remove_eval_traps — fingerprints and removes anti-debugging/timing trap eval calls.

Fingerprints eval payloads containing debugger loops and timing checks (such as
Date.now, setTimeout/unref, and threshold delta comparisons).

When a trap is identified, it removes the eval call and automatically climbs
up to remove any dedicated surrounding wrappers:
    (function () {
        try {
            (function () {
                (0, eval)('...trap payload...');
            }());
        } catch (e) {}
    }());
"""

import re
from jsir import Session, walk, node as make

_DEBUGGER_RE = re.compile(r'\bdebugger\b')

_TIMING_MARKERS = (
    'Date.now',
    'new Date',
    '_dateObj',
    'getTime',
    'valueOf',
)

_SCHEDULER_MARKERS = (
    'setTimeout',
    'setInterval',
    'unref',
)


def _is_eval_callee(callee, session: Session) -> bool:
    """Matches `eval` or `(0, eval)` (SequenceExpression ending in eval)."""
    if callee is None:
        return False
    if callee.kind == 'SequenceExpression':
        exprs = callee.children_in('expressions')
        callee = exprs[-1] if exprs else callee
    return (
        callee.kind == 'Identifier'
        and callee.attr('name') == 'eval'
        and session.scopes.binding_of(callee) is None
    )


def _is_trap_payload(source: str) -> bool:
    """Checks whether the evaluated string matches the debugger/timing trap pattern."""
    if not isinstance(source, str):
        return False

    has_debugger = bool(_DEBUGGER_RE.search(source))
    has_timing = any(marker in source for marker in _TIMING_MARKERS)
    has_scheduler = any(marker in source for marker in _SCHEDULER_MARKERS)

    # 1. Full timing + debugger trap (e.g. eval('debugger;' + this) with Date & setTimeout)
    if has_debugger and (has_timing or has_scheduler):
        return True

    # 2. Timing loop trap without explicit debugger statement
    if has_timing and has_scheduler and ('> 20' in source or '> 997' in source or 't.unref' in source):
        return True

    # 3. Dedicated short debugger payload
    if has_debugger and len(source.strip()) < 100:
        return True

    return False


def _find_dedicated_wrapper(call_node):
    """
    Climbs upward from the eval statement. If the statement is the sole occupant
    of an inner IIFE, which in turn is the sole occupant of a try/catch or outer IIFE,
    returns the outermost dedicated statement so the entire wrapper is purged.
    """
    parent = call_node.parent
    if parent is None or parent.kind != 'ExpressionStatement':
        return None

    current = parent

    while True:
        p = current.parent
        if p is None or p.kind != 'BlockStatement':
            break

        stmts = p.children_in('body')
        if len(stmts) != 1 or stmts[0] is not current:
            break

        gp = p.parent
        if gp is None:
            break

        # Case A: Self-executing anonymous IIFE: `(function () { ... })()`
        if gp.kind in ('FunctionExpression', 'ArrowFunctionExpression'):
            call_expr = gp.parent
            if call_expr and call_expr.kind == 'CallExpression' and call_expr.get('callee') is gp:
                expr_stmt = call_expr.parent
                if expr_stmt and expr_stmt.kind == 'ExpressionStatement':
                    current = expr_stmt
                    continue
            break

        # Case B: TryStatement `try { ... } catch (e) {}` with an empty catch body
        elif gp.kind == 'TryStatement' and gp.get('block') is p:
            handler = gp.get('handler')
            finalizer = gp.get('finalizer')
            if finalizer is not None:
                break
            if handler is not None:
                h_body = handler.get('body')
                if h_body and len(h_body.children_in('body')) > 0:
                    break
            current = gp
            continue

        break

    return current


def remove_eval_traps(session: Session) -> None:
    print("--- Applying Transformer: remove_eval_traps ---")
    tree = session.tree

    traps_to_remove = []

    for call in walk(tree.root, kinds={'CallExpression'}):
        if not call.live:
            continue

        args = call.children_in('arguments')
        if len(args) != 1:
            continue

        if not _is_eval_callee(call.get('callee'), session):
            continue

        arg = args[0]
        payload = arg.attr('value') if arg.kind == 'Literal' else session.constant_of(arg)

        if isinstance(payload, str) and _is_trap_payload(payload):
            traps_to_remove.append(call)

    removed_count = 0
    for call in traps_to_remove:
        if not call.live:
            continue

        target_stmt = _find_dedicated_wrapper(call)

        if target_stmt and target_stmt.live:
            tree.remove(target_stmt)
            removed_count += 1
        elif call.parent and call.parent.kind == 'ExpressionStatement' and call.parent.live:
            tree.remove(call.parent)
            removed_count += 1
        elif call.live:
            tree.replace(call, make('Identifier', name='undefined'))
            removed_count += 1

    print(f" |-> Removed {removed_count} eval trap(s)")
