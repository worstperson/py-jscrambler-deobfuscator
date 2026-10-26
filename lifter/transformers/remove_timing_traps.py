"""
remove_timing_traps — strip empty timing-based debugger / deobfuscator loops:

    do {
    } while (new Date() - new Date() < -100);

These loops execute an impossible timing check designed to trip or lag
emulators and anti-tamper analyzers. Because the body is empty and the
condition is always false, the loop has no side effects and can be safely
deleted.
"""

from jsir import Session, walk


def unwrap(node):
    """Unwrap ParenthesizedExpressions and SequenceExpressions if present."""
    curr = node
    while curr:
        if curr.kind == 'SequenceExpression':
            exprs = curr.children_in('expressions')
            curr = exprs[-1] if exprs else None
        elif curr.kind == 'ParenthesizedExpression':
            curr = curr.get('expression')
        else:
            break
    return curr


def is_new_date(node) -> bool:
    """Matches: new Date()"""
    node = unwrap(node)
    if not node or node.kind != 'NewExpression':
        return False
    callee = unwrap(node.get('callee'))
    return callee is not None and callee.kind == 'Identifier' and callee.attr('name') == 'Date'


def is_date_subtraction(node) -> bool:
    """Matches: new Date() - new Date()"""
    node = unwrap(node)
    if not node or node.kind != 'BinaryExpression':
        return False
    if node.attr('operator') != '-':
        return False
    return is_new_date(node.get('left')) and is_new_date(node.get('right'))


def is_negative_number(node) -> bool:
    """Matches literal negative values like -100 (either folded Literal or UnaryExpression)."""
    node = unwrap(node)
    if not node:
        return False
    # Literal with negative value: Literal(-100.0)
    if node.kind == 'Literal':
        val = node.attr('value')
        return isinstance(val, (int, float)) and val < 0
    # Unary negation: - 100
    if node.kind == 'UnaryExpression' and node.attr('operator') == '-':
        arg = unwrap(node.get('argument'))
        if arg and arg.kind == 'Literal':
            val = arg.attr('value')
            return isinstance(val, (int, float)) and val > 0
    return False


def is_timing_test(test_node) -> bool:
    """Matches comparisons like: new Date() - new Date() < -100"""
    test_node = unwrap(test_node)
    if not test_node or test_node.kind != 'BinaryExpression':
        return False

    op = test_node.attr('operator')
    left = test_node.get('left')
    right = test_node.get('right')

    # Standard form: `new Date() - new Date() < -100`
    if op in ('<', '<=') and is_date_subtraction(left) and is_negative_number(right):
        return True

    # Inverted form: `-100 > new Date() - new Date()`
    if op in ('>', '>=') and is_negative_number(left) and is_date_subtraction(right):
        return True

    return False


def is_empty_body(body_node) -> bool:
    """Matches {} or ;"""
    if not body_node:
        return True
    if body_node.kind == 'EmptyStatement':
        return True
    if body_node.kind == 'BlockStatement':
        stmts = body_node.children_in('body')
        return len(stmts) == 0
    return False


def is_timing_trap_loop(stmt) -> bool:
    """Matches: do {} while (new Date() - new Date() < -100); and while variations."""
    if not stmt or stmt.kind not in ('DoWhileStatement', 'WhileStatement'):
        return False
    return is_empty_body(stmt.get('body')) and is_timing_test(stmt.get('test'))


def remove_timing_traps(session: Session) -> None:
    print("--- Applying Transformer: remove_timing_traps ---")
    tree = session.tree

    traps = [
        node for node in walk(tree.root, kinds={'DoWhileStatement', 'WhileStatement'})
        if is_timing_trap_loop(node)
    ]

    if not traps:
        print(" |-> No timing traps found.")
        return

    total_removed = 0
    for loop in traps:
        if not loop.live:
            continue
        tree.remove(loop)
        total_removed += 1

    print(f" |-> Removed {total_removed} timing trap(s).")
