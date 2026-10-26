"""
propagate_runtime_aliases — propagates alias assignments to session.runtime_object.

Handles both:
    var b7P = _global;
and:
    b7P = _global;

Rules:
  - If b7P is never reassigned to anything else, all reads are replaced with _global,
    and both the assignment and the declaration are removed.
  - If b7P is reassigned, only reads where the active reaching write is _global
    are propagated. If the assignment was part of a VariableDeclarator, the assignment
    (init) is cleared to leave the declaration (`var b7P;`).
"""

from jsir import Session, walk, node as make, clone
from jsir.schema import slots_of, TARGET
from transformers.helpers.ast_utils import unwrap_expression


def _in_target_position(node):
    """True if node is the write target (LHS of assignment, update argument, etc.)."""
    parent = node.parent
    if parent is None:
        return False
    for slot, _arity, role in slots_of(parent.kind):
        if slot == node.parent_slot and role == TARGET:
            return True
    if parent.kind == 'UnaryExpression' and parent.attr('operator') == 'delete':
        return True
    if parent.kind == 'Property' and parent.get('value') is node \
            and parent.parent is not None and parent.parent.kind == 'ObjectPattern':
        return True
    return False


def _pos(node):
    """Source position or node ID for sequential ordering."""
    if node is None:
        return 0
    if getattr(node, 'origin', None):
        return node.origin[0]
    return node.nid


def _is_effective_write(write_node):
    """Writes that actually store a value. `var x;` stores nothing."""
    parent = write_node.parent
    if parent is not None and parent.kind == 'VariableDeclarator' \
            and parent.get('id') is write_node and parent.get('init') is None:
        return False
    return True


def _remove_assignment_node(tree, assign_node):
    """Removes an AssignmentExpression statement or prunes it from a SequenceExpression."""
    parent = assign_node.parent
    if parent is None or not parent.live:
        return

    if parent.kind == 'ExpressionStatement':
        tree.remove(parent)
    elif parent.kind == 'SequenceExpression':
        exprs = parent.children_in('expressions')
        surviving = [e for e in exprs if e is not assign_node and e.live]
        if len(surviving) == 0:
            gp = parent.parent
            if gp and gp.kind == 'ExpressionStatement':
                tree.remove(gp)
            else:
                tree.replace(parent, make('Identifier', name='undefined'))
        elif len(surviving) == 1:
            tree.replace(parent, clone(surviving[0]))
        else:
            tree.remove(assign_node)


def _remove_declarator_node(tree, declarator):
    """Removes a VariableDeclarator, or the whole VariableDeclaration if it was the only one."""
    decl_node = declarator.parent
    if decl_node is None or not decl_node.live or decl_node.kind != 'VariableDeclaration':
        return

    declarators = decl_node.children.get('declarations', [])
    if len(declarators) <= 1:
        if decl_node.parent is not None:
            tree.remove(decl_node)
    else:
        tree.remove(declarator)


def propagate_runtime_aliases(session: Session) -> None:
    print("--- Applying Transformer: propagate_runtime_aliases ---")

    runtime_object = getattr(session, 'runtime_object', None)
    if not runtime_object:
        print(" |-> No runtime object found on session. Skipping.")
        return

    tree = session.tree
    scopes = session.scopes

    total_propagated = 0
    total_assignments_cleaned = 0

    while True:
        # Collect candidate assignments: `b7P = _global` or `var b7P = _global`
        # Map: binding -> list of (assignment_node, target_ident, is_declarator)
        bindings_to_global_writes = {}

        for node in walk(tree.root, kinds={'VariableDeclarator', 'AssignmentExpression'}):
            if not node.live:
                continue

            target_id = None
            source_expr = None
            is_decl = False

            if node.kind == 'VariableDeclarator':
                target_id = node.get('id')
                source_expr = unwrap_expression(node.get('init'))
                is_decl = True
            elif node.kind == 'AssignmentExpression' and node.attr('operator', '=') == '=':
                target_id = unwrap_expression(node.get('left'))
                source_expr = unwrap_expression(node.get('right'))
                is_decl = False

            if not (target_id and target_id.kind == 'Identifier'):
                continue
            if not (source_expr and source_expr.kind == 'Identifier'):
                continue
            if source_expr.attr('name') != runtime_object:
                continue

            binding = scopes.binding_of(target_id)
            if not binding:
                scope = scopes.scope_of(target_id)
                binding = scope.lookup(target_id.attr('name')) if scope else None

            if not binding:
                continue

            bindings_to_global_writes.setdefault(binding, []).append((node, target_id, is_decl))

        if not bindings_to_global_writes:
            break

        round_propagated = 0

        for binding, global_write_tuples in bindings_to_global_writes.items():
            # 1. Determine all effective writes for this binding
            all_writes = list(getattr(binding, 'writes', ()))
            effective_writes = [w for w in all_writes if w.live and _is_effective_write(w)]
            global_write_idents = {target_id for _, target_id, _ in global_write_tuples}

            # Check if there are other writes besides assignments to _global
            reassigned = any(w not in global_write_idents for w in effective_writes)

            # 2. Propagate reads
            reads_to_replace = []
            for r in getattr(binding, 'reads', ()):
                if not r.live or _in_target_position(r):
                    continue

                if not reassigned:
                    # Never reassigned: every read is guaranteed to be _global
                    reads_to_replace.append(r)
                else:
                    # Reassigned: find the last effective write before this read
                    r_pos = _pos(r)
                    prior_writes = [w for w in effective_writes if _pos(w) < r_pos]
                    if prior_writes:
                        prior_writes.sort(key=_pos)
                        latest_write = prior_writes[-1]
                        if latest_write in global_write_idents:
                            reads_to_replace.append(r)

            for r in reads_to_replace:
                if r.live:
                    tree.replace(r, make('Identifier', name=runtime_object))
                    round_propagated += 1

            # 3. Clean up the assignments/declarations according to reassignment status
            for assign_node, _target_id, is_decl in global_write_tuples:
                if not assign_node.live:
                    continue

                if is_decl:
                    if reassigned:
                        # Remove assignment, leave declaration: `var b7P;`
                        tree.set_slot(assign_node, 'init', None)
                        total_assignments_cleaned += 1
                    else:
                        # Remove both declaration and assignment
                        _remove_declarator_node(tree, assign_node)
                        total_assignments_cleaned += 1
                else:
                    # Standalone assignment: remove the statement
                    _remove_assignment_node(tree, assign_node)
                    total_assignments_cleaned += 1

                    # If not reassigned and a separate uninitialized declaration exists, remove it too
                    if not reassigned and binding.declaration and binding.declaration.live:
                        if binding.declaration.kind == 'VariableDeclarator':
                            _remove_declarator_node(tree, binding.declaration)

        total_propagated += round_propagated
        if round_propagated == 0:
            break

        scopes.mark_dirty()

    print(f" |-> Propagated {total_propagated} read(s) to {runtime_object}, "
          f"cleaned up {total_assignments_cleaned} assignment(s)/declaration(s).")
