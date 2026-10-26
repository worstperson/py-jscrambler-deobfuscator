"""
restore_local_variables — restores `arguments` and named parameter references.
"""

from jsir import Session, walk, node as make
from transformers.helpers.ast_utils import references_name
from jsir.schema import FUNCTION_KINDS


def restore_local_variables(session: Session) -> None:
    print("--- Applying Transformer: restore_local_variables ---")

    tree = session.tree
    index = session.scopes

    plans = []
    for fn in walk(tree.root, kinds=FUNCTION_KINDS):
        plan = _analyze_function(index, fn)
        if plan is not None:
            plans.append(plan)

    for plan in plans:
        _apply_plan(tree, plan)

    restore_argument_references(session)


def _analyze_function(index, fn):
    """Pure read: resolve the array-builder pattern and decide what every
    occurrence turns into, without touching the tree. Returns None if `fn`
    doesn't match the pattern."""
    body = fn.get('body')
    if body is None or body.kind != 'BlockStatement':
        return None

    pattern = _find_arguments_array_pattern(body)
    if pattern is None:
        return None

    var_name = pattern['var_name']
    binding = index.scope_of(fn).lookup(var_name)
    if binding is None:
        return None

    owner_fn = _nearest_trustworthy_function(fn)

    occurrences = []   # (node, replacement_node)
    sites = []          # (node, idx) for every T[idx] occurrence, idx > 0
    for node in walk(fn, kinds={'MemberExpression'}):
        if not node.attr('computed', False):
            continue
        obj = node.get('object')
        prop = node.get('property')
        if obj is None or prop is None:
            continue
        if obj.kind != 'Identifier' or obj.attr('name') != var_name:
            continue
        if index.binding_of(obj) is not binding:
            continue
        if prop.kind != 'Literal':
            continue
        idx = prop.attr('value')
        if not isinstance(idx, (int, float)) or isinstance(idx, bool):
            continue
        idx = int(idx)

        if idx == 0:
            arguments_id = make('Identifier', name='arguments')
            arguments_id.attrs['owner_fn'] = owner_fn
            occurrences.append((node, arguments_id))
        else:
            sites.append((node, idx))

    written_indices = {idx for node, idx in sites if _is_write_target(node)}
    for node, idx in sites:
        if idx in written_indices:
            occurrences.append((node, make('Identifier', name=f'{var_name}{idx}')))
        else:
            arguments_id = make('Identifier', name='arguments')
            arguments_id.attrs['owner_fn'] = owner_fn
            replacement = make('MemberExpression', object=arguments_id,
                               property=make('Literal', value=float(idx)), computed=True)
            occurrences.append((node, replacement))

    return {
        'pattern': pattern,
        'occurrences': occurrences,
        'written_indices': written_indices,
    }


def _is_write_target(node):
    """True if `node` (a T[idx] MemberExpression) is the target of an assignment."""
    parent = node.parent
    return (parent is not None and parent.kind == 'AssignmentExpression' and
            parent.get('left') is node)


def _apply_plan(tree, plan):
    for node, replacement in plan['occurrences']:
        if node.live:
            tree.replace(node, replacement)

    var_name = plan['pattern']['var_name']
    init_decl = plan['pattern']['init_decl']
    declarator = plan['pattern']['declarator']
    init_assign = plan['pattern']['init_assign']
    written_indices = plan['written_indices']

    if declarator.live:
        siblings = init_decl.children_in('declarations')
        if written_indices:
            new_declarators = [
                make('VariableDeclarator',
                     id=make('Identifier', name=f'{var_name}{idx}'), init=None)
                for idx in sorted(written_indices)
            ]
            if len(siblings) == 1:
                hoisted_decl = make('VariableDeclaration', kind='var',
                                    declarations=new_declarators)
                if init_decl.live:
                    tree.replace(init_decl, hoisted_decl)
            else:
                tree.insert_after(declarator, new_declarators)
                tree.remove(declarator)
        else:
            if len(siblings) == 1:
                if init_decl.live:
                    tree.remove(init_decl)
            else:
                tree.remove(declarator)

    if init_assign is not None and init_assign.live:
        tree.remove(init_assign)


def _find_arguments_array_pattern(body):
    """Find `var T = [arguments]` or `var T; ... T = [arguments]` at the top of a
    function body in O(statements) without quadratic AST walks."""
    stmts = body.children_in('body')

    # Case 1: Direct init `var T = [arguments]`
    for stmt in stmts:
        if stmt.kind == 'VariableDeclaration':
            for declarator in stmt.children_in('declarations'):
                init = declarator.get('init')
                if init is not None and _is_arguments_array(init):
                    id_node = declarator.get('id')
                    if id_node is not None and id_node.kind == 'Identifier':
                        return {'var_name': id_node.attr('name'), 'init_decl': stmt,
                                'declarator': declarator, 'init_assign': None}

    # Case 2: Delayed init `var T; ... T = [arguments]`
    for i, stmt in enumerate(stmts):
        if stmt.kind == 'ExpressionStatement':
            expr = stmt.get('expression')
            if expr is not None and expr.kind == 'AssignmentExpression':
                left, right = expr.get('left'), expr.get('right')
                if left is not None and left.kind == 'Identifier' and right is not None and _is_arguments_array(right):
                    var_name = left.attr('name')
                    for prev in stmts[:i]:
                        if prev.kind == 'VariableDeclaration':
                            for decl in prev.children_in('declarations'):
                                d_id = decl.get('id')
                                if d_id is not None and d_id.kind == 'Identifier' and d_id.attr('name') == var_name and decl.get('init') is None:
                                    used_between = any(references_name(s, var_name) for s in stmts[stmts.index(prev) + 1:i])
                                    if not used_between:
                                        return {'var_name': var_name, 'init_decl': prev,
                                                'declarator': decl, 'init_assign': stmt}
                    break

    return None


def _is_arguments_array(node):
    if node.kind != 'ArrayExpression':
        return False
    elements = node.children_in('elements')
    return (len(elements) == 1 and elements[0].kind == 'Identifier' and
            elements[0].attr('name') == 'arguments')


# ---------------------------------------------------------------------------
# Step 2: arguments[i] → param_name
# ---------------------------------------------------------------------------

def _is_arguments_base(node):
    """Returns True if `node` is either an Identifier named 'arguments'
    or the expression `[arguments][0]`."""
    if node is None:
        return False
    if node.kind == 'Identifier' and node.attr('name') == 'arguments':
        return True
    if node.kind == 'MemberExpression' and node.attr('computed', False):
        obj = node.get('object')
        prop = node.get('property')
        if obj is not None and obj.kind == 'ArrayExpression':
            elements = obj.children_in('elements')
            if len(elements) == 1 and elements[0].kind == 'Identifier' and elements[0].attr('name') == 'arguments':
                if prop is not None and prop.kind == 'Literal' and prop.attr('value') == 0:
                    return True
    return False


def restore_argument_references(session: Session) -> None:
    """Replace `arguments[i]` (and `[arguments][0][i]`) with the named parameter from the enclosing
    function, skipping arrow functions (they inherit `arguments`) and
    functions invoked via `.apply(ctx, arguments)` forwarding."""
    tree = session.tree

    replacements = []
    for node in walk(tree.root, kinds={'MemberExpression'}):
        if not node.attr('computed', False):
            continue
        obj, prop = node.get('object'), node.get('property')
        if obj is None or prop is None:
            continue
        if not _is_arguments_base(obj):
            continue
        if prop.kind != 'Literal':
            continue
        idx = prop.attr('value')
        if not isinstance(idx, (int, float)) or isinstance(idx, bool):
            continue
        idx = int(idx)

        owner_fn = obj.attrs.get('owner_fn') if obj.kind == 'Identifier' else None
        if owner_fn is not None:
            fn = owner_fn if owner_fn.live else None
        else:
            fn = _enclosing_real_function(node)
        if fn is None:
            continue
        params = fn.children_in('params')
        if idx < 0 or idx >= len(params):
            continue
        param = params[idx]
        if param.kind != 'Identifier':
            continue
        replacements.append((node, param.attr('name')))

    for node, name in replacements:
        if node.live:
            tree.replace(node, make('Identifier', name=name))


def _is_apply_arguments_forwarding(fn):
    """True if `fn` is the object of `<fn>.apply(ctx, arguments)`."""
    if fn.kind != 'FunctionExpression':
        return False
    parent = fn.parent
    if parent is None or parent.kind != 'MemberExpression':
        return False
    if parent.attr('computed', False) or parent.get('object') is not fn:
        return False
    prop = parent.get('property')
    if prop is None or prop.kind != 'Identifier' or prop.attr('name') != 'apply':
        return False
    call = parent.parent
    if call is None or call.kind != 'CallExpression' or call.get('callee') is not parent:
        return False
    call_args = call.children_in('arguments')
    return (len(call_args) == 2 and call_args[1].kind == 'Identifier' and
            call_args[1].attr('name') == 'arguments')


def _nearest_trustworthy_function(fn):
    """Starting at `fn` (inclusive), walk outward past arrow functions
    and functions invoked via `.apply(ctx, arguments)` forwarding."""
    current = fn
    while current is not None:
        if current.kind == 'ArrowFunctionExpression' or _is_apply_arguments_forwarding(current):
            current = current.enclosing(*FUNCTION_KINDS)
            continue
        return current
    return None


def _enclosing_real_function(node):
    fn = node.enclosing(*FUNCTION_KINDS)
    return _nearest_trustworthy_function(fn) if fn is not None else None
