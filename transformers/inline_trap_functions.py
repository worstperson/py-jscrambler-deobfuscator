"""
inline_trap_functions — inlines the body of anti-tamper trap submodules into call sites.

Transforms:
    _global[434878] = function () {
        return {
            z0b1UQ3: function () {
                do {
                } while (true);
            }
        };
    }();
    ...
    var g4v = () => {
        _global[434878].z0b1UQ3();
        if (s9N++ > 0)
            s9N--;
    };

Into:
    var g4v = () => {
        do {
        } while (true);
        if (s9N++ > 0)
            s9N--;
    };
"""

from jsir import Session, walk, node as make, clone
from jsir.schema import FUNCTION_KINDS
from transformers.helpers.ast_utils import unwrap_expression
from transformers.helpers.globals import is_runtime_object, get_member_key
from transformers.find_runtime_functions import (
    find_method_returned_object,
    get_method_node_for_property,
)


def _is_trap_body(fn_node):
    """Checks if a function contains an infinite loop or trap construct."""
    if not fn_node or fn_node.kind not in FUNCTION_KINDS:
        return False
    body = fn_node.get('body')
    if not body:
        return False
    for loop in walk(body, kinds={'WhileStatement', 'DoWhileStatement', 'ForStatement'}):
        test = loop.get('test')
        if test and test.kind == 'Literal' and test.attr('value') is True:
            return True
        if loop.kind == 'ForStatement' and loop.get('test') is None and loop.get('update') is None:
            return True
    return False


def _extract_trap_statements(method_fn):
    """Extracts the statements of the trap method, stripping any trailing empty return."""
    body = method_fn.get('body')
    if not body:
        return []
    if body.kind != 'BlockStatement':
        return [make('ExpressionStatement', expression=clone(body))]

    stmts = list(body.children_in('body'))
    if stmts and stmts[-1].kind == 'ReturnStatement':
        arg = stmts[-1].get('argument')
        if arg is None or (arg.kind == 'Identifier' and arg.attr('name') == 'undefined') or \
                (arg.kind == 'Literal' and arg.attr('value') is None):
            stmts.pop()
    return stmts


def _match_trap_call(call_node, session, global_key, method_name):
    """Matches `_global[key].method(...)` or `_global[key].method.call/apply(...)`."""
    if call_node.kind != 'CallExpression':
        return False

    callee = unwrap_expression(call_node.get('callee'))
    if not callee:
        return False

    # Handle .call(...) and .apply(...)
    if callee.kind == 'MemberExpression':
        prop = get_member_key(callee)
        if prop in ('call', 'apply'):
            callee = unwrap_expression(callee.get('object'))
            if not callee:
                return False

    if callee.kind != 'MemberExpression':
        return False

    if str(get_member_key(callee)) != str(method_name):
        return False

    container = unwrap_expression(callee.get('object'))
    if not container or container.kind != 'MemberExpression':
        return False

    if not is_runtime_object(container.get('object'), session):
        return False

    c_key = get_member_key(container)
    return c_key is not None and str(c_key) == str(global_key)


def _inline_call_site(tree, call_node, trap_statements):
    """Replaces call_node with the inlined trap statements."""
    cloned_stmts = [clone(s) for s in trap_statements]
    parent = call_node.parent

    # Case 1: Standard statement context: `_global[key].method();`
    if parent and parent.kind == 'ExpressionStatement':
        stmt = parent
        container = stmt.parent.children.get(stmt.parent_slot)
        if isinstance(container, list):
            tree.replace_many(stmt, cloned_stmts)
        elif not cloned_stmts:
            tree.replace(stmt, make('EmptyStatement'))
        elif len(cloned_stmts) == 1:
            tree.replace(stmt, cloned_stmts[0])
        else:
            tree.replace(stmt, make('BlockStatement', body=cloned_stmts))
        return

    # Case 2: Expression context (e.g. sequence): wrap in an IIFE to preserve syntax
    iife = make(
        'CallExpression',
        callee=make(
            'ArrowFunctionExpression',
            params=[],
            body=make('BlockStatement', body=cloned_stmts),
            isAsync=False,
            generator=False,
            expression=False,
        ),
        arguments=[],
        optional=False,
    )
    tree.replace(call_node, iife)


def _collect_trap_definitions(session):
    """Collects trap definitions from session.runtime_entries or by AST inspection."""
    traps = []

    # Source 1: Check session.runtime_entries if find_runtime_functions already ran
    entries = getattr(session, 'runtime_entries', None)
    if entries:
        for entry in entries:
            if entry.get('label') != 'trap_call':
                continue

            method_fn = entry.get('method_fn')
            target_node, obj_node = find_method_returned_object(method_fn)
            methods = entry.get('method_names', [])

            for m_name in methods:
                fn_node = get_method_node_for_property(
                    method_fn, target_node, obj_node, m_name, session
                )
                if fn_node:
                    traps.append({
                        'global_key': entry['global_key'],
                        'method_name': m_name,
                        'method_node': fn_node,
                        'assign_node': entry.get('assign_node'),
                        'runtime_entry': entry,
                    })

    # Source 2: Direct AST inspection fallback (e.g. if run before find_runtime_functions)
    if not traps:
        tree = session.tree
        for assign in walk(tree.root, kinds={'AssignmentExpression'}):
            if assign.attr('operator') != '=':
                continue
            left = unwrap_expression(assign.get('left'))
            if not left or left.kind != 'MemberExpression':
                continue
            if not is_runtime_object(left.get('object'), session):
                continue

            global_key = get_member_key(left)
            if global_key is None:
                continue

            right = unwrap_expression(assign.get('right'))
            if not right or right.kind != 'CallExpression':
                continue
            callee = unwrap_expression(right.get('callee'))
            if not callee or callee.kind not in FUNCTION_KINDS:
                continue

            target_node, obj_node = find_method_returned_object(callee)
            if not obj_node or obj_node.kind != 'ObjectExpression':
                continue

            for prop in obj_node.children_in('properties'):
                if prop.kind not in ('Property', 'ObjectProperty'):
                    continue
                key_node = prop.get('key')
                prop_name = (
                    key_node.attr('name')
                    if key_node and key_node.kind == 'Identifier'
                    else key_node.attr('value') if key_node and key_node.kind == 'Literal' else None
                )
                if not prop_name:
                    continue

                fn_val = unwrap_expression(prop.get('value'))
                if fn_val and fn_val.kind in FUNCTION_KINDS and _is_trap_body(fn_val):
                    traps.append({
                        'global_key': global_key,
                        'method_name': prop_name,
                        'method_node': fn_val,
                        'assign_node': assign,
                        'runtime_entry': None,
                    })

    return traps


def inline_trap_functions(session: Session) -> None:
    print("--- Applying Transformer: inline_trap_functions ---")

    tree = session.tree
    traps = _collect_trap_definitions(session)
    if not traps:
        print(" |-> No trap definitions found.")
        return

    total_inlined = 0

    for trap in traps:
        key = trap['global_key']
        method_name = trap['method_name']
        method_node = trap['method_node']
        trap_stmts = _extract_trap_statements(method_node)

        # 1. Find and inline all calls to `_global[key].method(...)`
        call_sites = []
        for call in walk(tree.root, kinds={'CallExpression'}):
            if not call.live:
                continue
            if _match_trap_call(call, session, key, method_name):
                call_sites.append(call)

        inlined_here = 0
        for call in call_sites:
            if call.live:
                _inline_call_site(tree, call, trap_stmts)
                inlined_here += 1

        total_inlined += inlined_here

        print(f" |-> Trap _global[{key!r}].{method_name}: inlined at {inlined_here} call site(s).")

    print(f" |-> Total: {total_inlined} call(s) inlined.")
