"""
remove_runtime_submodules — strips all global runtime submodule assignments
discovered and classified in `session.runtime_entries`, provided that no
active calls to any of their methods remain in the AST outside of the runtime
submodules themselves.

Removes the entire assignment statement:
    _global[key] = (function () { ... })();
    _global.key = function () { ... }();
"""

from jsir import Session, walk, clone, node as make
from jsir.schema import FUNCTION_KINDS
from transformers.find_runtime_functions import (
    is_runtime_object,
    get_member_key,
    unwrap_expression,
)


def is_descendant_of_any(node, ancestor_set: set) -> bool:
    """True if node is inside any of the AST subtrees in ancestor_set."""
    curr = node
    while curr is not None:
        if curr in ancestor_set:
            return True
        curr = curr.parent
    return False


def count_runtime_method_calls(session: Session, global_key, method_names: list[str], runtime_nodes: set) -> int:
    """
    Counts live CallExpressions calling `_global[global_key].method(...)`
    that occur outside of ALL runtime submodule subtrees.
    """
    tree = session.tree
    key_str = str(global_key)
    method_set = {str(m) for m in method_names} if method_names else None
    call_count = 0

    for call in walk(tree.root, kinds={'CallExpression'}):
        if not call.live:
            continue

        # Ignore calls occurring inside ANY runtime submodule definition
        if runtime_nodes and is_descendant_of_any(call, runtime_nodes):
            continue

        callee = unwrap_expression(call.get('callee'))
        if not callee:
            continue

        # Handle `.call(...)` and `.apply(...)` invocations
        if callee.kind == 'MemberExpression':
            prop_key = get_member_key(callee)
            if prop_key in ('call', 'apply'):
                inner = unwrap_expression(callee.get('object'))
                if inner and inner.kind == 'MemberExpression':
                    callee = inner

        if callee.kind != 'MemberExpression':
            continue

        # callee is `container.method`
        container = unwrap_expression(callee.get('object'))
        if not container or container.kind != 'MemberExpression':
            continue

        # container.object must be the runtime object
        if not is_runtime_object(container.get('object'), session):
            continue

        # container.property must match this runtime's key
        c_key = get_member_key(container)
        if c_key is None or str(c_key) != key_str:
            continue

        # Check if the called property matches any of this runtime's methods
        method_key = get_member_key(callee)
        if method_set is not None:
            if method_key is not None and str(method_key) in method_set:
                call_count += 1
        else:
            # If no specific method names known, any call on the container counts
            call_count += 1

    return call_count


def is_numeric_literal(node) -> bool:
    """True if node is a literal number (including negative/unary numbers)."""
    if not node:
        return False
    if node.kind == 'Literal':
        val = node.attr('value')
        return isinstance(val, (int, float)) and not isinstance(val, bool)
    if node.kind == 'UnaryExpression' and node.attr('operator') in ('+', '-'):
        return is_numeric_literal(node.get('argument'))
    return False


def check_is_runtime_object(node, session: Session, runtime_object: str | None) -> bool:
    """Matches direct references to the runtime object identifier or alias."""
    if not node:
        return False
    node = unwrap_expression(node)
    if not node:
        return False
    if runtime_object and node.kind == 'Identifier' and node.attr('name') == runtime_object:
        return True
    return is_runtime_object(node, session)


def is_global_member_access(node, session: Session, runtime_object: str | None) -> bool:
    """
    Matches any property access on _global:
      - `_global[540022]` (bracket notation / computed)
      - `_global.O7FF` (dot notation / non-computed)
    """
    if not node:
        return False
    node = unwrap_expression(node)
    if not node or node.kind != 'MemberExpression':
        return False
    obj = unwrap_expression(node.get('object'))
    return check_is_runtime_object(obj, session, runtime_object)


def is_global_access(node, session: Session, runtime_object: str | None) -> bool:
    """
    Matches any access to _global:
      - direct reference: `_global`
      - property access: `_global[540022]` or `_global.O7FF`
    """
    if not node:
        return False
    node = unwrap_expression(node)
    if not node:
        return False
    return (
        check_is_runtime_object(node, session, runtime_object)
        or is_global_member_access(node, session, runtime_object)
    )


def is_string_literal(node) -> bool:
    """True if node is a literal string."""
    if not node:
        return False
    node = unwrap_expression(node)
    if not node:
        return False
    return node.kind == 'Literal' and isinstance(node.attr('value'), str)


def is_boolean_literal(node) -> bool:
    """True if node is a literal string."""
    if not node:
        return False
    node = unwrap_expression(node)
    if not node:
        return False
    return node.kind == 'Literal' and isinstance(node.attr('value'), bool)


def is_target_assignment(node, session: Session, runtime_object: str) -> bool:
    """
    Matches assignments to _global properties:
      - `_global.N3 = 45` / `_global[9] = 34` (numeric literals)
      - `_global[540022] = 'NS0'` / `_global.prop = 'NS0'` (string literals)
      - `_global[63743] = _global[284166]` / `_global.prop = _global.other` (global member access)
      - `_global.O7FF = _global` / `_global[key] = _global` (direct global assignment)
    """
    if not node or node.kind != 'AssignmentExpression':
        return False
    if node.attr('operator') != '=':
        return False

    left = unwrap_expression(node.get('left'))
    # Left side must be a property access on _global (_global[prop] or _global.prop)
    if not is_global_member_access(left, session, runtime_object):
        return False

    right = unwrap_expression(node.get('right'))
    if not right:
        return False

    # 1. Number literal (e.g. 45, -1)
    if is_numeric_literal(right):
        return True

    # 2. String literal (e.g. 'NS0')
    if is_string_literal(right):
        return True

    # 3. Direct _global or _global property access (_global, _global[284166], _global.O7FF)
    if is_global_access(right, session, runtime_object):
        return True

    if  is_boolean_literal(right):
        return True

    return False


def get_base_statements(runtime_root):
    """Fetches direct, top-level statements of runtime_root without entering child blocks."""
    if not runtime_root:
        return ()
    if runtime_root.kind in FUNCTION_KINDS:
        body = runtime_root.get('body')
        if body and body.kind == 'BlockStatement':
            return body.children_in('body')
    elif runtime_root.kind in ('Program', 'BlockStatement'):
        return runtime_root.children_in('body')
    return ()


def get_function_name(node) -> str | None:
    """Extracts function identifier name from FunctionDeclaration or VariableDeclarator."""
    if not node:
        return None
    if node.kind in ('FunctionDeclaration', 'FunctionExpression'):
        fn_id = node.get('id')
        if fn_id and fn_id.kind == 'Identifier':
            return fn_id.attr('name')
    elif node.kind == 'VariableDeclarator':
        var_id = node.get('id')
        if var_id and var_id.kind == 'Identifier':
            return var_id.attr('name')
    return None


def is_string_fn_assignment(expr, session: Session, runtime_object: str, string_fn_names: set[str]) -> bool:
    """Matches `_global.prop = d1p` or `_global[prop] = d1p` where d1p is a string function."""
    if not expr or expr.kind != 'AssignmentExpression':
        return False
    if expr.attr('operator') != '=':
        return False

    left = unwrap_expression(expr.get('left'))
    if not is_global_member_access(left, session, runtime_object):
        return False

    right = unwrap_expression(expr.get('right'))
    if not right or right.kind != 'Identifier':
        return False

    return right.attr('name') in string_fn_names


def remove_runtime_submodules(session: Session, exclude_labels: set[str] | None = None) -> None:
    """
    Removes all runtime submodule definitions stored in session.runtime_entries,
    provided that no active calls to any of their methods remain in the AST
    outside of the runtime submodules themselves.
    """
    print("--- Applying Transformer: remove_runtime_submodules ---")

    tree = session.tree
    entries = getattr(session, 'runtime_entries', None)
    if entries:
        removed_count = 0

        # Collect all runtime submodule assignment nodes slated for removal so we can
        # ignore internal cross-submodule method calls between them.
        candidate_runtime_nodes = {
            entry['assign_node'] for entry in entries
            if entry.get('assign_node') and entry['assign_node'].live
            and not (exclude_labels and entry.get('label') in exclude_labels)
        }

        for entry in entries:
            label = entry.get('label', 'unknown')
            key = entry.get('global_key', '?')
            methods = entry.get('method_names', [])

            # 1. Optional label filter
            if exclude_labels and label in exclude_labels:
                print(f" |-> Skipping _global[{key!r}] ({label}) due to exclude filter.")
                continue

            assign_node = entry.get('assign_node')
            if not assign_node or not assign_node.live:
                continue


            # 2. Check if there are active external calls (ignoring all runtime submodules)
            active_calls = count_runtime_method_calls(session, key, methods, candidate_runtime_nodes)
            if active_calls > 0:
                print(f" |-> Keeping _global[{key!r}] ({label}): {active_calls} active call(s) still remaining.")
                continue

            parent = assign_node.parent
            if not parent or not parent.live:
                continue

            # 3. Standard top-level statement: `_global[key] = (function () { ... })();`
            if parent.kind == 'ExpressionStatement':
                tree.remove(parent)
                removed_count += 1

            # 4. Inside a comma sequence: `( _global[key] = (...)(), realCode() )`
            elif parent.kind == 'SequenceExpression':
                surviving = [e for e in parent.children_in('expressions') if e is not assign_node and e.live]
                if len(surviving) == 1:
                    tree.replace(parent, clone(surviving[0]))
                elif len(surviving) == 0:
                    grandparent = parent.parent
                    if grandparent and grandparent.kind == 'ExpressionStatement':
                        tree.remove(grandparent)
                    else:
                        tree.replace(parent, make('Identifier', name='undefined'))
                else:
                    tree.remove(assign_node)
                removed_count += 1

            # 5. Fallback for any other expression context
            else:
                tree.replace(assign_node, make('Identifier', name='undefined'))
                removed_count += 1

            print(f" |-> Removed runtime submodule: _global[{key!r}] ({label})")

        print(f" |-> Successfully removed {removed_count} runtime submodule(s).")

    # -----------------------------------------------------------------------
    # Setup aliases function cleanup
    # -----------------------------------------------------------------------
    fn = getattr(session, 'alias_table_function', None)
    if fn and getattr(fn, 'live', True):
        fn_id = fn.get('id')
        fn_name = fn_id.attr('name') if fn_id and fn_id.kind == 'Identifier' else None

        stmt = fn.enclosing_statement()
        if stmt and stmt.live:
            tree.remove(stmt)
            removed_calls = 0
            if fn_name:
                for call in walk(tree.root, kinds={'CallExpression'}):
                    if not call.live:
                        continue
                    callee = call.get('callee')
                    if callee and callee.kind == 'Identifier' and callee.attr('name') == fn_name:
                        parent = call.parent
                        if not parent or not parent.live:
                            continue

                        # Case A: Standalone expression statement
                        if parent.kind == 'ExpressionStatement':
                            tree.remove(parent)
                            removed_calls += 1

                        # Case B: Inside a SequenceExpression, e.g. (_global = ..., f7BiEU(_global))
                        elif parent.kind == 'SequenceExpression':
                            surviving = [e for e in parent.children_in('expressions') if e is not call and e.live]
                            if len(surviving) == 1:
                                tree.replace(parent, clone(surviving[0]))
                            elif len(surviving) == 0:
                                grandparent = parent.parent
                                if grandparent and grandparent.kind == 'ExpressionStatement':
                                    tree.remove(grandparent)
                            else:
                                tree.set_slot(parent, 'expressions', surviving)
                            removed_calls += 1

            if removed_calls > 0:
                print(f" |-> Removed setup aliases function definition '{fn_name or '<anonymous>'}' and removed {removed_calls} invocation call")
            else:
                print(f" |-> Removed setup aliases function definition '{fn_name or '<anonymous>'}'")

    # -----------------------------------------------------------------------
    # String functions cleanup
    # -----------------------------------------------------------------------
    string_nodes = getattr(session, 'string_functions', ())
    string_fn_names = {
        name for node in string_nodes
        if (name := get_function_name(node)) is not None
    }

    runtime_object = getattr(session, 'runtime_object', None)
    runtime_root = getattr(session, 'runtime_root', session.tree.root)

    # 1. Look for base-level assignments: `_global.D4u05sk = d1p;` or `_global[15093] = d1p;`
    if runtime_object and string_fn_names:
        base_stmts = get_base_statements(runtime_root)
        assignments_to_remove = []

        for stmt in base_stmts:
            if not stmt or not stmt.live or stmt.kind != 'ExpressionStatement':
                continue
            expr = stmt.get('expression')
            if not expr:
                continue

            # Standard statement: `_global.D4u05sk = d1p;`
            if is_string_fn_assignment(expr, session, runtime_object, string_fn_names):
                assignments_to_remove.append((stmt, 'full'))

            # Inside a comma sequence: `(_global.D4u05sk = d1p, ...)`
            elif expr.kind == 'SequenceExpression':
                exprs = expr.children_in('expressions')
                if any(is_string_fn_assignment(e, session, runtime_object, string_fn_names) for e in exprs):
                    assignments_to_remove.append((stmt, 'sequence'))

        for stmt, mode in reversed(assignments_to_remove):
            if not stmt.live:
                continue
            if mode == 'full':
                tree.remove(stmt)
                print(" |-> Removed string function")
            elif mode == 'sequence':
                expr = stmt.get('expression')
                exprs = expr.children_in('expressions')
                surviving = [e for e in exprs if not is_string_fn_assignment(e, session, runtime_object, string_fn_names)]
                if len(surviving) == 0:
                    tree.remove(stmt)
                elif len(surviving) == 1:
                    tree.replace(expr, clone(surviving[0]))
                else:
                    tree.set_slot(expr, 'expressions', surviving)
                print(" |-> Removed string function assignment from sequence")

    # 2. Remove the actual string function declarations
    for node in string_nodes:
        if getattr(node, 'live', True):
            print(" |-> Removing string data")
            tree.remove(node)

    # -----------------------------------------------------------------------
    # VM function cleanup
    # -----------------------------------------------------------------------

    vm_function = getattr(session, 'vm_function', None)
    if vm_function and getattr(vm_function, 'live', True):
        session.tree.remove(vm_function)
        session.dce_keep = [] # FIXME

    # -----------------------------------------------------------------------
    # Factory stub function cleanup
    # -----------------------------------------------------------------------
    runtime_stub = getattr(session, 'runtime_stub', None)
    if runtime_stub and getattr(runtime_stub, 'live', True):
        print(" |-> Removing runtime stub function")
        tree.remove(runtime_stub)

    # -----------------------------------------------------------------------
    # Global getter cleanup (only remove if no other runtime_object refs remain)
    # -----------------------------------------------------------------------
    getter = getattr(session, 'global_getter', None)
    runtime_object = getattr(session, 'runtime_object', None)

    if getter and getattr(getter, 'live', True):
        getter_scope = {getter}
        if getter.parent and getter.parent.kind == 'ExpressionStatement':
            getter_scope.add(getter.parent)

        has_other_references = False
        if runtime_object:
            count = 0
            for id_node in walk(tree.root, kinds={'Identifier'}):
                if not id_node.live:
                    continue
                if id_node.attr('name') == runtime_object:
                    if not is_descendant_of_any(id_node, getter_scope):
                        has_other_references = True
                        break
                    else:
                        count += 1

        if not has_other_references:
            print(f" |-> Removing global getter (no remaining '{runtime_object}' references)")
            target = getter.parent if (getter.parent and getter.parent.kind == 'ExpressionStatement' and getter.parent.live) else getter
            tree.remove(target)
        else:
            print(f" |-> Keeping global getter: '{runtime_object}' is still referenced elsewhere")
