"""
propagate_runtime_globals — performs a flat walk on `runtime_root` to:
  1. Extract and propagate global self-aliases: `_global.O7FF = _global`
  2. Extract and inline literal constants: `_global.N3 = 45`
  3. Extract and inline decryptor call expressions: `_global.g9P = _global.o2q.h4MXf1y(868)`
  4. Extract and rewrite member-to-member aliases: `_global[413254] = _global[434878]`
  5. Remove the assignment statements from `runtime_root`.
"""

from jsir import Session, walk, clone, node as make
from jsir.schema import FUNCTION_KINDS

from transformers.helpers.ast_utils import unwrap_expression
from transformers.helpers.literals import (
    extract_literal_value,
    normalize_value_to_key,
    value_to_node,
)
from transformers.helpers.globals import (
    is_global_or_alias,
    get_member_key,
)


def get_normalized_member_key(node, session: Session, global_aliases: set[str], constants_table: dict) -> str | None:
    """Extracts and normalizes the property key of a MemberExpression,
    resolving identifiers and nested global lookups through the constants table.
    """
    if not node or node.kind != 'MemberExpression':
        return None
    prop = node.get('property')
    if not prop:
        return None

    if not node.attr('computed', False):
        return prop.attr('name') if prop.kind == 'Identifier' else None

    prop = unwrap_expression(prop)
    if not prop:
        return None

    val, ok = extract_literal_value(prop)
    if ok:
        return normalize_value_to_key(val)

    if prop.kind == 'Identifier':
        name = prop.attr('name')
        if constants_table and name in constants_table:
            return normalize_value_to_key(constants_table[name])
        return name

    # Handle nested global lookups: `_global[_global[63743]]`
    if prop.kind == 'MemberExpression':
        obj = unwrap_expression(prop.get('object'))
        if is_global_or_alias(obj, session, global_aliases):
            inner_key = get_normalized_member_key(prop, session, global_aliases, constants_table)
            if inner_key and inner_key in constants_table:
                return normalize_value_to_key(constants_table[inner_key])

    return None


def is_assignment_target(node) -> bool:
    parent = node.parent
    if not parent:
        return False
    if parent.kind == 'AssignmentExpression' and parent.get('left') is node:
        return True
    if parent.kind == 'UpdateExpression' and parent.get('argument') is node:
        return True
    if parent.kind in ('ForInStatement', 'ForOfStatement') and parent.get('left') is node:
        return True
    if parent.kind == 'UnaryExpression' and parent.attr('operator') == 'delete' and parent.get('argument') is node:
        return True
    return False


def is_descendant_of_any(node, ancestor_set: set) -> bool:
    curr = node.parent
    while curr is not None:
        if curr in ancestor_set:
            return True
        curr = curr.parent
    return False


def clean_up_assignments(tree, assignments_to_remove: set) -> int:
    removed_count = 0
    seq_parents = set()
    expr_stmts = set()

    for assign in assignments_to_remove:
        if not assign.live:
            continue
        p = assign.parent
        if p and p.kind == 'SequenceExpression':
            seq_parents.add(p)
        elif p and p.kind == 'ExpressionStatement':
            expr_stmts.add(p)

    for seq in seq_parents:
        if not seq.live:
            continue
        exprs = seq.children_in('expressions')
        surviving = [e for e in exprs if e not in assignments_to_remove]
        removed_count += (len(exprs) - len(surviving))
        if len(surviving) == 0:
            p = seq.parent
            if p and p.kind == 'ExpressionStatement':
                tree.remove(p)
            else:
                tree.replace(seq, make('Identifier', name='undefined'))
        elif len(surviving) == 1:
            tree.replace(seq, clone(surviving[0]))
        else:
            tree.set_slot(seq, 'expressions', surviving)

    for stmt in expr_stmts:
        if not stmt.live:
            continue
        expr = stmt.get('expression')
        if expr in assignments_to_remove:
            tree.remove(stmt)
            removed_count += 1

    for assign in assignments_to_remove:
        if assign.live:
            tree.replace(assign, make('Identifier', name='undefined'))
            removed_count += 1

    return removed_count


# FIXME we need to be able to reapply this
def propagate_runtime_globals(session: Session) -> None:
    print("--- Applying Transformer: propagate_runtime_globals ---")

    tree = session.tree
    runtime_object = getattr(session, 'runtime_object', None)
    runtime_root = getattr(session, 'runtime_root', tree.root)

    constants_table = {}
    calls_table = {}
    member_aliases = {}
    global_aliases = set()
    assignments_to_remove = set()

    changed = True
    while changed:
        changed = False

        for assign in walk(runtime_root, kinds={'AssignmentExpression'}, enter_functions=False):
            if not assign.live or assign in assignments_to_remove:
                continue
            if assign.attr('operator') != '=':
                continue

            left = unwrap_expression(assign.get('left'))
            if not left or left.kind != 'MemberExpression':
                continue

            obj = unwrap_expression(left.get('object'))
            if not is_global_or_alias(obj, session, global_aliases):
                continue

            key = get_normalized_member_key(left, session, global_aliases, constants_table)
            if key is None:
                continue

            right = unwrap_expression(assign.get('right'))
            if not right:
                continue

            # Case A: Global self-alias (`_global.O7FF = _global`)
            if is_global_or_alias(right, session, global_aliases):
                if key not in global_aliases:
                    global_aliases.add(key)
                    changed = True
                assignments_to_remove.add(assign)
                continue

            # Case B: Literal constant (`_global.N3 = 45`)
            val, ok = extract_literal_value(right)
            if ok:
                if key not in constants_table or constants_table[key] != val:
                    constants_table[key] = val
                    changed = True
                assignments_to_remove.add(assign)
                continue

            # Case C: Decryptor call expression (`_global.g9P = _global.o2q.h4MXf1y(868)`)
            if right.kind == 'CallExpression':
                callee = unwrap_expression(right.get('callee'))
                if callee and callee.kind not in FUNCTION_KINDS:
                    calls_table[key] = right
                    assignments_to_remove.add(assign)
                    changed = True
                    continue

            # Case D: Chained lookup (`_global[63743] = _global[284166]`)
            if right.kind == 'MemberExpression':
                r_obj = unwrap_expression(right.get('object'))
                if is_global_or_alias(r_obj, session, global_aliases):
                    r_key = get_normalized_member_key(right, session, global_aliases, constants_table)
                    if r_key:
                        target_key = r_key
                        visited = {key}
                        while target_key in member_aliases and target_key not in visited:
                            visited.add(target_key)
                            target_key = member_aliases[target_key]

                        if target_key in global_aliases:
                            if key not in global_aliases:
                                global_aliases.add(key)
                                changed = True
                            assignments_to_remove.add(assign)
                        elif target_key in constants_table:
                            r_val = constants_table[target_key]
                            if key not in constants_table or constants_table[key] != r_val:
                                constants_table[key] = r_val
                                changed = True
                            assignments_to_remove.add(assign)
                        elif target_key in calls_table:
                            calls_table[key] = calls_table[target_key]
                            assignments_to_remove.add(assign)
                            changed = True
                        else:
                            if member_aliases.get(key) != target_key:
                                member_aliases[key] = target_key
                                changed = True
                            assignments_to_remove.add(assign)

    print(
        f" |-> Discovered {len(constants_table)} literal(s), {len(calls_table)} call(s), "
        f"{len(member_aliases)} member alias(es), and {len(global_aliases)} global alias(es)."
    )

    replaced_constants = 0
    replaced_calls = 0
    replaced_member_aliases = 0
    replaced_aliases = 0

    for _ in range(5):
        replacements = []
        scheduled_targets = set()

        for member in walk(tree.root, kinds={'MemberExpression'}):
            if not member.live or member in scheduled_targets:
                continue

            if is_descendant_of_any(member, assignments_to_remove) or is_assignment_target(member):
                continue

            obj = unwrap_expression(member.get('object'))
            if not is_global_or_alias(obj, session, global_aliases):
                continue

            key = get_normalized_member_key(member, session, global_aliases, constants_table)
            if key is None:
                continue

            resolved_key = key
            visited = set()
            while resolved_key in member_aliases and resolved_key not in visited:
                visited.add(resolved_key)
                resolved_key = member_aliases[resolved_key]

            if resolved_key in constants_table:
                val = constants_table[resolved_key]
                replacements.append((member, value_to_node(val), 'constant'))
                scheduled_targets.add(member)

            elif resolved_key in calls_table:
                call_node = clone(calls_table[resolved_key])
                replacements.append((member, call_node, 'call'))
                scheduled_targets.add(member)

            elif resolved_key in global_aliases:
                target_name = runtime_object
                replacements.append((member, make('Identifier', name=target_name), 'alias'))
                scheduled_targets.add(member)

            elif resolved_key != key:
                prop_node = make('Literal', value=int(resolved_key)) if resolved_key.isdigit() else make('Literal', value=resolved_key)
                new_member = make('MemberExpression', object=clone(obj), property=prop_node, computed=True)
                replacements.append((member, new_member, 'member_alias'))
                scheduled_targets.add(member)

        if not replacements:
            break

        filtered_replacements = [
            (target, rep, kind) for target, rep, kind in replacements
            if not is_descendant_of_any(target, scheduled_targets)
        ]

        for target, rep, kind in filtered_replacements:
            if not target.live:
                continue
            tree.replace(target, rep)
            if kind == 'constant':
                replaced_constants += 1
            elif kind == 'call':
                replaced_calls += 1
            elif kind == 'member_alias':
                replaced_member_aliases += 1
            else:
                replaced_aliases += 1

    print(
        f" |-> Inlined {replaced_constants} constant(s), {replaced_calls} call(s), "
        f"{replaced_member_aliases} member alias(es), and unwrapped {replaced_aliases} global alias(es)."
    )

    removed_count = clean_up_assignments(tree, assignments_to_remove)
    print(f" |-> Removed {removed_count} global assignment statement(s).")

    if hasattr(session, 'oracle'):
        if hasattr(session.oracle, 'members'):
            for k, v in constants_table.items():
                session.oracle.members[f'globalThis.{k}'] = v
            for k, target in member_aliases.items():
                if target in constants_table:
                    session.oracle.members[f'globalThis.{k}'] = constants_table[target]
        if hasattr(session.oracle, 'global_aliases'):
            for alias in global_aliases:
                session.oracle.global_aliases.add(alias)
