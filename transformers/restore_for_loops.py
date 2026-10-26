"""
restore_for_loops — Reconstruct ForStatements from WhileStatements with trailing updates.

Transforms patterns such as:
    var i = 0;
    while (i < 10) {
        ...
        i++;
    }

Into:
    var i;
    for (i = 0; i < 10; i++) {
        ...
    }

Also handles:
  - Removal of preceding inline assignments (`i = 0; while (...)`).
  - Validation of `continue` statements within the loop: if a `continue` was
    preceded by an equivalent update (to ensure the loop counter stepped before
    continuing), that update is eliminated because the reconstructed `for` loop
    handles it automatically in its update clause.
"""

from jsir import Session, walk, node as make, clone

_LOOP_KINDS = frozenset((
    'WhileStatement',
    'DoWhileStatement',
    'ForStatement',
    'ForInStatement',
    'ForOfStatement',
))


def _is_numeric_literal(node) -> bool:
    if node is None or node.kind != 'Literal':
        return False
    val = node.attr('value')
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def _is_equivalent_update(node1, node2) -> bool:
    if node1 is None or node2 is None or node1.kind != node2.kind:
        return False

    if node1.kind == 'UpdateExpression':
        arg1, arg2 = node1.get('argument'), node2.get('argument')
        if (arg1 is None or arg2 is None or
                arg1.kind != 'Identifier' or arg2.kind != 'Identifier'):
            return False
        return (
            node1.attr('operator') == node2.attr('operator') and
            arg1.attr('name') == arg2.attr('name')
        )

    if node1.kind == 'AssignmentExpression':
        if node1.attr('operator') != node2.attr('operator'):
            return False

        left1, left2 = node1.get('left'), node2.get('left')
        if (left1 is None or left2 is None or
                left1.kind != 'Identifier' or left2.kind != 'Identifier'):
            return False
        if left1.attr('name') != left2.attr('name'):
            return False

        right1, right2 = node1.get('right'), node2.get('right')
        if right1 is None or right2 is None:
            return False

        if _is_numeric_literal(right1) and _is_numeric_literal(right2):
            return right1.attr('value') == right2.attr('value')

        if right1.kind == 'Identifier' and right2.kind == 'Identifier':
            return right1.attr('name') == right2.attr('name')

    return False


def restore_for_loops(session: Session) -> None:
    print("--- Applying Transformer: restore_for_loops ---")
    tree = session.tree
    count = 0

    for while_node in list(walk(tree.root, kinds={'WhileStatement'})):
        if not while_node.live:
            continue

        body = while_node.get('body')
        if body is None or body.kind != 'BlockStatement':
            continue

        body_statements = list(body.children_in('body'))
        if not body_statements:
            continue

        last_stmt = body_statements[-1]
        if last_stmt.kind != 'ExpressionStatement':
            continue

        last_expr = last_stmt.get('expression')
        if last_expr is None:
            continue

        loop_var_name = None
        update_node = None

        if last_expr.kind == 'UpdateExpression':
            arg = last_expr.get('argument')
            if arg is not None and arg.kind == 'Identifier':
                loop_var_name = arg.attr('name')
                update_node = last_expr
        elif last_expr.kind == 'AssignmentExpression':
            left = last_expr.get('left')
            if left is not None and left.kind == 'Identifier':
                loop_var_name = left.attr('name')
                update_node = last_expr

        if not loop_var_name or update_node is None:
            continue

        # -------------------------------------------------------------
        # Step 1: Scan preceding siblings for initialization
        # -------------------------------------------------------------
        init_action = None
        parent = while_node.parent

        if parent is not None:
            siblings = parent.children.get(while_node.parent_slot)
            idx = while_node.index_in_parent()
            if isinstance(siblings, list) and idx is not None:
                for i in range(idx - 1, -1, -1):
                    candidate = siblings[i]
                    has_ref = False
                    has_unsafe_ref = False
                    valid_init_expr = None
                    is_var_decl = False
                    target_decl = None

                    for id_node in walk(candidate, kinds={'Identifier'}):
                        if id_node.attr('name') == loop_var_name:
                            has_ref = True
                            p = id_node.parent

                            if p is not None and p.kind == 'VariableDeclarator' and p.get('id') is id_node:
                                if _is_numeric_literal(p.get('init')):
                                    is_var_decl = True
                                    target_decl = p
                                else:
                                    has_unsafe_ref = True
                            elif (p is not None and p.kind == 'AssignmentExpression' and
                                  p.get('left') is id_node and p.attr('operator', '=') == '='):
                                if _is_numeric_literal(p.get('right')):
                                    valid_init_expr = p
                                else:
                                    has_unsafe_ref = True
                            else:
                                has_unsafe_ref = True

                    if has_ref:
                        if not has_unsafe_ref:
                            if is_var_decl and target_decl is not None:
                                init_action = ('var_decl', target_decl)
                            elif valid_init_expr is not None:
                                init_action = ('expr_stmt', candidate, valid_init_expr)
                        break

        # -------------------------------------------------------------
        # Step 2: Validate continue statement paths
        # -------------------------------------------------------------
        can_transform = True
        stmts_to_remove = [last_stmt]

        for cont_node in walk(while_node, kinds={'ContinueStatement'}):
            parent_loop = cont_node.enclosing(*_LOOP_KINDS)
            if parent_loop is not while_node:
                continue
            if cont_node.get('label') is not None:
                continue

            c_parent = cont_node.parent
            if c_parent is None:
                can_transform = False
                break

            c_siblings = c_parent.children.get(cont_node.parent_slot)
            if not isinstance(c_siblings, list):
                can_transform = False
                break

            c_idx = cont_node.index_in_parent()
            if c_idx is None or c_idx == 0:
                can_transform = False
                break

            prev_stmt = c_siblings[c_idx - 1]
            if (prev_stmt.kind == 'ExpressionStatement' and
                    _is_equivalent_update(prev_stmt.get('expression'), update_node)):
                stmts_to_remove.append(prev_stmt)
            else:
                can_transform = False
                break

        if not can_transform:
            continue

        # -------------------------------------------------------------
        # Step 3: Apply transformations safely
        # -------------------------------------------------------------
        cloned_update = clone(update_node)
        test_expr = while_node.get('test')
        cloned_test = clone(test_expr) if test_expr is not None else None

        # Remove dead update statements from loop body
        for stmt in dict.fromkeys(stmts_to_remove):
            if stmt.live:
                tree.remove(stmt)

        # Build `init` node and clean up original declaration / statement
        init_node = None
        if init_action is not None:
            if init_action[0] == 'var_decl':
                target_decl = init_action[1]
                init_node = make(
                    'AssignmentExpression',
                    operator='=',
                    left=make('Identifier', name=loop_var_name),
                    right=clone(target_decl.get('init'))
                )
                tree.set_slot(target_decl, 'init', None)
            elif init_action[0] == 'expr_stmt':
                stmt_to_remove, valid_expr = init_action[1], init_action[2]
                init_node = clone(valid_expr)
                if stmt_to_remove.live:
                    tree.remove(stmt_to_remove)

        cloned_body = clone(while_node.get('body'))
        for_stmt = make(
            'ForStatement',
            init=init_node,
            test=cloned_test,
            update=cloned_update,
            body=cloned_body,
        )

        tree.replace(while_node, for_stmt)
        count += 1

    print(f" |-> Restored {count} for loop(s)")
