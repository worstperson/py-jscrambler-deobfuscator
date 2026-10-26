"""
unnest_assignment_expressions — Extract embedded assignments from expressions.

Transforms:
    return (e = expr1).replace(expr2);
Into:
    e = expr1;
    return e.replace(expr2);
"""

from jsir import Session, walk, node as make, clone

_STATEMENT_LIST_SLOTS = frozenset({
    ('Program', 'body'),
    ('BlockStatement', 'body'),
    ('SwitchCase', 'consequent'),
})

_SINGLE_STATEMENT_SLOTS = frozenset({
    ('IfStatement', 'consequent'),
    ('IfStatement', 'alternate'),
    ('WhileStatement', 'body'),
    ('DoWhileStatement', 'body'),
    ('ForStatement', 'body'),
    ('ForInStatement', 'body'),
    ('ForOfStatement', 'body'),
})


def _find_enclosing_statement(node):
    """
    Climb up past all expressions until reaching a real statement.
    Returns (statement_node, is_in_statement_list).
    """
    curr = node
    while curr is not None:
        parent = curr.parent
        if parent is None:
            return None, False

        slot = curr.parent_slot
        # 1. Statement directly inside a block/program list
        if (parent.kind, slot) in _STATEMENT_LIST_SLOTS:
            return curr, True

        # 2. Statement in a braceless body (e.g. if (c) return ...;)
        if (parent.kind, slot) in _SINGLE_STATEMENT_SLOTS:
            return curr, False

        curr = parent

    return None, False


def _is_short_circuit_guarded(node, stmt):
    """
    Safety check: ensure `node` isn't inside a short-circuiting branch (&&, ||, ?:)
    between `node` and `stmt`, which would make hoisting before `stmt` unsound.
    """
    curr = node
    while curr is not None and curr is not stmt:
        parent = curr.parent
        if parent is None:
            break
        if parent.kind == 'LogicalExpression' and curr.parent_slot == 'right':
            return True
        if parent.kind == 'ConditionalExpression' and curr.parent_slot in ('consequent', 'alternate'):
            return True
        curr = parent
    return False


def unnest_assignment_expressions(session: Session) -> None:
    print("--- Applying Transformer: unnest_assignment_expressions ---")
    tree = session.tree
    count = 0

    for assign in list(walk(tree.root, kinds={'AssignmentExpression'})):
        if not assign.live or assign.attr('operator', '=') != '=':
            continue

        parent = assign.parent
        if parent is None:
            continue

        # Skip standalone statements like `x = 1;`
        if parent.kind == 'ExpressionStatement' and parent.get('expression') is assign:
            continue

        # Target `(x = ...).property`
        if parent.kind == 'MemberExpression' and parent.get('object') is assign:
            target_var = assign.get('left')
            if target_var is None or target_var.kind != 'Identifier':
                continue

            stmt, in_list = _find_enclosing_statement(assign)
            if stmt is None or not stmt.live:
                continue

            # Don't hoist if guarded by short-circuiting logic (e.g. a && (b = c).d)
            if _is_short_circuit_guarded(assign, stmt):
                continue

            # 1. Create `x = <rhs>;` to hoist before the enclosing statement
            hoisted_stmt = make('ExpressionStatement', expression=clone(assign))

            # 2. Replace `(x = <rhs>)` inside the member expression with just `x`
            tree.replace(assign, clone(target_var))

            # 3. Splice `hoisted_stmt` before `stmt` safely
            if in_list:
                tree.replace_many(stmt, [hoisted_stmt, clone(stmt)])
            else:
                # Handles single-statement contexts like `if (c) return (e = 1).x;`
                tree.replace(stmt, make('BlockStatement', body=[hoisted_stmt, clone(stmt)]))

            count += 1

    print(f" |-> Total embedded assignments unnested: {count}")
