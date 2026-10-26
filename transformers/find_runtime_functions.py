from jsir import Session, walk, node as make
from jsir.schema import FUNCTION_KINDS

from transformers.helpers.ast_utils import unwrap_expression
from transformers.helpers.globals import (
    is_runtime_object,
    get_member_key,
    resolve_identifier,
)


# ---------------------------------------------------------------------------
# Structural helpers
# ---------------------------------------------------------------------------

def is_equivalent(n1, n2) -> bool:
    if n1 is n2:
        return True
    if not (n1 and n2) or n1.kind != n2.kind:
        return False

    if n1.kind == 'Identifier':
        return n1.attr('name') == n2.attr('name')

    if n1.kind == 'Literal':
        return n1.attr('value') == n2.attr('value')

    if n1.kind == 'MemberExpression':
        if n1.attr('computed', False) != n2.attr('computed', False):
            return False
        return (
            is_equivalent(n1.get('object'), n2.get('object')) and
            is_equivalent(n1.get('property'), n2.get('property'))
        )

    return False


def extract_key_representation(node) -> str | int | float | None:
    if not node:
        return None
    node = unwrap_expression(node)
    if not node:
        return None

    if node.kind == 'Literal':
        val = node.attr('value')
        if isinstance(val, (str, int, float)) and not isinstance(val, bool):
            return val
        return str(val)

    if node.kind == 'Identifier':
        return node.attr('name')

    if node.kind == 'MemberExpression':
        obj_key = extract_key_representation(node.get('object'))
        prop_key = extract_key_representation(node.get('property'))
        if obj_key is not None and prop_key is not None:
            if node.attr('computed', False):
                return f"{obj_key}[{prop_key!r}]" if isinstance(prop_key, str) else f"{obj_key}[{prop_key}]"
            else:
                return f"{obj_key}.{prop_key}"

    return None


def resolve_object_expression(node, fn_node):
    """Traces node back to ObjectExpression or returns node as target container."""
    if not node:
        return None

    node = unwrap_expression(node)
    if not node:
        return None

    if node.kind == 'ObjectExpression':
        return node

    if node.kind == 'AssignmentExpression':
        return resolve_object_expression(node.get('right'), fn_node)

    if node.kind in ('Identifier', 'MemberExpression'):
        for assign in walk(fn_node, kinds={'AssignmentExpression'}):
            left = assign.get('left')
            if left and is_equivalent(left, node):
                obj = resolve_object_expression(assign.get('right'), fn_node)
                if obj:
                    return obj

        if node.kind == 'Identifier':
            var_name = node.attr('name')
            for decl in walk(fn_node, kinds={'VariableDeclarator'}):
                d_id = decl.get('id')
                if d_id and d_id.kind == 'Identifier' and d_id.attr('name') == var_name:
                    obj = resolve_object_expression(decl.get('init'), fn_node)
                    if obj:
                        return obj

    return node


def find_method_returned_object(fn_node):
    """Returns tuple of (target_node, resolved_obj_node)."""
    if not fn_node or fn_node.kind not in FUNCTION_KINDS:
        return None, None

    body = fn_node.get('body')
    if body and body.kind == 'ObjectExpression':
        return body, body

    for ret in walk(fn_node, kinds={'ReturnStatement'}):
        enclosing_fn = ret.enclosing(*FUNCTION_KINDS)
        if enclosing_fn is not fn_node:
            continue

        arg = ret.get('argument')
        if not arg:
            continue

        target_node = unwrap_expression(arg)
        obj_node = resolve_object_expression(target_node, fn_node)
        return target_node, obj_node

    return None, None


def get_object_property_names(target_node, obj_node, fn_node) -> list[str]:
    """Extracts property keys from the ObjectExpression and top-level assignments."""
    keys = []
    seen = set()

    if obj_node and obj_node.kind == 'ObjectExpression':
        props = obj_node.get('properties') or obj_node.children_in('properties') or []
        for prop in props:
            if prop.kind in ('Property', 'ObjectProperty'):
                key_node = prop.get('key')
                if not key_node:
                    continue
                key_val = extract_key_representation(key_node)
                if key_val is not None and str(key_val) not in seen:
                    keys.append(str(key_val))
                    seen.add(str(key_val))

    if fn_node:
        for assign in walk(fn_node, kinds={'AssignmentExpression'}):
            if assign.enclosing(*FUNCTION_KINDS) is not fn_node:
                continue

            left = assign.get('left')
            if not left or left.kind != 'MemberExpression':
                continue

            obj = left.get('object')
            if (target_node and is_equivalent(obj, target_node)) or (obj_node and is_equivalent(obj, obj_node)):
                prop_node = left.get('property')
                if left.attr('computed', False):
                    key_val = extract_key_representation(prop_node)
                else:
                    key_val = prop_node.attr('name') if prop_node and prop_node.kind == 'Identifier' else extract_key_representation(prop_node)

                if key_val is not None and str(key_val) not in seen:
                    keys.append(str(key_val))
                    seen.add(str(key_val))

    return keys


# ---------------------------------------------------------------------------
# Call-site helpers (used exclusively for domain lock folding & cf_env_test)
# ---------------------------------------------------------------------------

def _is_global_container(node, session, global_key):
    node = unwrap_expression(node)
    if not node or node.kind != 'MemberExpression':
        return False
    if not is_runtime_object(node.get('object'), session):
        return False
    return get_member_key(node) == global_key


def _strip_index_chain_upward(call):
    current = call
    depth = 0
    parent = current.parent
    while parent is not None and parent.kind == 'MemberExpression' and \
            parent.get('object') is current and parent.attr('computed', False):
        current = parent
        depth += 1
        parent = current.parent
    return current, depth


def find_method_calls(tree, session, global_key, method_name):
    refs, calls = [], []
    for member in walk(tree.root, kinds={'MemberExpression'}):
        if get_member_key(member) != method_name:
            continue
        if not _is_global_container(member.get('object'), session, global_key):
            continue
        refs.append(member)
        parent = member.parent
        if parent is not None and parent.kind == 'CallExpression' and parent.get('callee') is member:
            calls.append(parent)
    return calls, refs


def _classify_call_context(call, session, global_key, method_name):
    current, chain_depth = _strip_index_chain_upward(call)
    parent = current.parent
    context = {'chain_depth': chain_depth, 'kind': 'value_use'}

    if parent is None:
        return context

    if parent.kind == 'UnaryExpression' and parent.attr('operator') == '!':
        grandparent = parent.parent
        if grandparent is not None and grandparent.kind == 'IfStatement' and grandparent.get('test') is parent:
            context['kind'] = 'if_falsy'
            return context

    elif parent.kind == 'IfStatement' and parent.get('test') is current:
        context['kind'] = 'if_truthy'
        return context

    '''elif parent.kind == 'BinaryExpression' and (parent.get('left') is current or parent.get('right') is current):
        op = parent.attr('operator')
        if op in ('==', '===', '!==', '!='):
            context['kind'] = 'equality_compare'
            return context
        elif op in ('<', '<=', '>', '>='):
            context['kind'] = 'relational_compare'
            return context

    elif parent.kind == 'ConditionalExpression':
        context['kind'] = 'conditional_expression'
        return context'''

    return context


def resolve_cf_env_test(session, global_key, method_name) -> str:
    calls, _ = find_method_calls(session.tree, session, global_key, method_name)
    if not calls:
        return 'cf_env_test_truthy'

    contexts = [_classify_call_context(c, session, global_key, method_name) for c in calls]
    kinds = [ctx['kind'] for ctx in contexts]

    if 'if_falsy' in kinds or 'if_truthy' in kinds:
        return 'cf_env_test_falsy' if kinds.count('if_falsy') >= kinds.count('if_truthy') else 'cf_env_test_truthy'

    return 'cf_env_test_truthy'


# ---------------------------------------------------------------------------
# AST Signature extraction
# ---------------------------------------------------------------------------

def _resolve_last_assigned_value(identifier_node, session: Session = None, seen=None):
    if not (identifier_node and identifier_node.kind == 'Identifier' and session and hasattr(session, 'scopes')):
        return identifier_node
    if seen is None:
        seen = set()
    if identifier_node.nid in seen:
        return identifier_node
    seen.add(identifier_node.nid)

    scopes = session.scopes
    binding = scopes.binding_of(identifier_node) or scopes.scope_of(identifier_node).lookup(identifier_node.attr('name'))
    if not binding:
        return identifier_node

    if binding.declaration and binding.declaration.kind in FUNCTION_KINDS:
        return binding.declaration

    def _pos(n):
        return n.origin[0] if n.origin else n.nid

    ref_pos = _pos(identifier_node)
    candidates = []

    if binding.declaration and binding.declaration.kind == 'VariableDeclarator':
        init = binding.declaration.get('init')
        target = binding.target or binding.declaration.get('id')
        if init and target and _pos(target) < ref_pos:
            candidates.append((_pos(target), init))

    for write in getattr(binding, 'writes', ()):
        if write is identifier_node:
            continue
        write_pos = _pos(write)
        if write_pos < ref_pos:
            parent = write.parent
            if parent and parent.kind == 'AssignmentExpression' and parent.get('left') is write:
                assigned_val = parent.get('right')
                if assigned_val:
                    candidates.append((write_pos, assigned_val))

    if not candidates:
        return identifier_node

    candidates.sort(key=lambda item: item[0])
    return unwrap_expression(candidates[-1][1])


def _classify_return_node(node, session: Session = None):
    if not node:
        return 'void'
    node = unwrap_expression(node)
    if not node:
        return 'void'

    if node.kind == 'Identifier' and session:
        resolved = _resolve_last_assigned_value(node, session)
        if resolved and resolved is not node:
            return _classify_return_node(resolved, session)

    if node.kind == 'Literal':
        val = node.attr('value')
        if isinstance(val, bool):
            return 'bool'
        if isinstance(val, (int, float)):
            return 'number'
        if isinstance(val, str):
            return 'string'
        if val is None:
            return 'null'
        return 'literal'

    return node.kind


def resolve_to_function(node, session: Session = None, seen=None):
    if not node:
        return None
    if seen is None:
        seen = set()
    if node.nid in seen:
        return None
    seen.add(node.nid)

    node = unwrap_expression(node)
    if not node:
        return None

    if node.kind in FUNCTION_KINDS:
        return node

    if node.kind == 'CallExpression':
        callee = unwrap_expression(node.get('callee'))
        fn = resolve_to_function(callee, session, seen)
        if fn and fn.kind in FUNCTION_KINDS:
            return fn

    if node.kind == 'Identifier' and session and hasattr(session, 'scopes'):
        scopes = session.scopes
        binding = scopes.binding_of(node) or scopes.scope_of(node).lookup(node.attr('name'))
        if binding:
            decl = binding.declaration

            if decl and decl.kind in FUNCTION_KINDS:
                return decl

            if decl and decl.kind == 'VariableDeclarator' and decl.get('init'):
                fn = resolve_to_function(decl.get('init'), session, seen)
                if fn and fn.kind in FUNCTION_KINDS:
                    return fn

            for write in getattr(binding, 'writes', ()):
                parent = write.parent
                if parent and parent.kind == 'AssignmentExpression' and parent.get('left') is write:
                    right = parent.get('right')
                    if right:
                        fn = resolve_to_function(right, session, seen)
                        if fn and fn.kind in FUNCTION_KINDS:
                            return fn

    return node


def get_method_node_for_property(callee_fn, target_node, obj_node, prop_name, session: Session = None):
    if obj_node and obj_node.kind == 'ObjectExpression':
        props = obj_node.get('properties') or obj_node.children_in('properties') or []
        for prop in props:
            if prop.kind in ('Property', 'ObjectProperty'):
                key_node = prop.get('key')
                if extract_key_representation(key_node) == prop_name:
                    return resolve_to_function(prop.get('value'), session)

    if callee_fn:
        for assign in walk(callee_fn, kinds={'AssignmentExpression'}):
            left = assign.get('left')
            if left and left.kind == 'MemberExpression':
                if (target_node and is_equivalent(left.get('object'), target_node)) or \
                   (obj_node and is_equivalent(left.get('object'), obj_node)):
                    if get_member_key(left) == prop_name:
                        return resolve_to_function(assign.get('right'), session)

    return None


def extract_method_signature(method_node, session: Session = None):
    method_node = resolve_to_function(method_node, session)

    if not method_node or method_node.kind not in FUNCTION_KINDS:
        return {'params': 0, 'returns': ['None']}

    raw_params = method_node.get('params') or method_node.children_in('params') or []
    param_count = len(raw_params)

    returns = []
    body = method_node.get('body')

    if body and body.kind != 'BlockStatement':
        returns.append(_classify_return_node(body, session))
    elif body and body.kind == 'BlockStatement':
        for ret in walk(body, kinds={'ReturnStatement'}):
            if ret.enclosing(*FUNCTION_KINDS) is not method_node:
                continue
            returns.append(_classify_return_node(ret.get('argument'), session))

    if not returns:
        returns.append('None')

    return {
        'params': param_count,
        'returns': returns
    }


# ---------------------------------------------------------------------------
# Dual-method order-independent matching
# ---------------------------------------------------------------------------

def classify_dual_methods(m1, sig1, m2, sig2):
    """Matches dual-method modules in either order.

    Returns (label, {method_name: role_name}).
    """
    p1, r1 = sig1['params'], sig1['returns']
    p2, r2 = sig2['params'], sig2['returns']

    def is_match(p, r, target_p, target_r):
        if p != target_p:
            return False
        if target_r == 'None':
            return r in (['None'], [])
        if isinstance(target_r, (list, tuple)):
            return r == list(target_r)
        return len(r) == 1 and r[0] == target_r

    def test_pair(pA, rA, roleA, pB, rB, roleB):
        if is_match(p1, r1, pA, rA) and is_match(p2, r2, pB, rB):
            return {m1: roleA, m2: roleB}
        if is_match(p2, r2, pA, rA) and is_match(p1, r1, pB, rB):
            return {m2: roleA, m1: roleB}
        return None

    # 1. State Machine: Calculator (0 params, BinaryExpression) & Setter (1 param, void)
    roles = test_pair(0, 'BinaryExpression', 'calculator', 1, 'None', 'setter')
    if roles:
        return 'state_machine', roles

    # 2. Toggle Gates: Toggle Gate (0 params, AssignmentExpression) & Random Gate (0 params, LogicalExpression)
    roles = test_pair(0, 'AssignmentExpression', 'toggle_gate', 0, 'LogicalExpression', 'random_gate')
    if roles:
        return 'toggle_gates', roles

    # 3. Domain Lock: Both accept 4 parameters
    roles = test_pair(4, 'MemberExpression', 'cached_validator', 4, 'CallExpression', 'hostname_validator')
    if roles:
        return 'domain_lock', roles

    # 4. Internal Storage: Getter (1 param, MemberExpression) & Setter (2 params, void)
    roles = test_pair(1, 'MemberExpression', 'getter', 2, 'None', 'setter')
    if roles:
        return 'internal_storage', roles

    # 5. Integrity Monitor: Register (2 params, CallExpression) & Teardown (0 params, void)
    roles = test_pair(2, 'CallExpression', 'register', 0, 'None', 'teardown')
    if roles:
        return 'integrity_monitor', roles

    # Fallback for unrecognized dual-method modules
    return 'unknown', {m1: 'unknown', m2: 'unknown'}


# ---------------------------------------------------------------------------
# Entry point: fingerprint factories, then classify each one
# ---------------------------------------------------------------------------

def find_runtime_functions(session: Session) -> None:
    print("--- Applying Transformer: find_runtime_functions ---")

    runtime_object = session.runtime_object
    if not runtime_object:
        print(" |-> No runtime object found.")
        return

    tree = session.tree
    matched_entries = []

    for assign in walk(tree.root, kinds={'AssignmentExpression'}):
        if assign.attr('operator') != '=':
            continue

        left = assign.get('left')
        if not left or left.kind != 'MemberExpression':
            continue

        obj = left.get('object')
        if not is_runtime_object(obj, session):
            continue

        global_key = get_member_key(left)
        if global_key is None:
            continue

        right = unwrap_expression(assign.get('right'))
        if not right:
            continue

        callee = None
        if right.kind == 'CallExpression':
            callee = unwrap_expression(right.get('callee'))
        elif right.kind in FUNCTION_KINDS:
            callee = right

        if not callee:
            continue

        target_node, obj_expr = find_method_returned_object(callee)
        if target_node is not None:
            prop_names = get_object_property_names(target_node, obj_expr, callee)
            if prop_names or (obj_expr and obj_expr.kind == 'ObjectExpression'):
                matched_entries.append({
                    'assign_node': assign,
                    'global_key': global_key,
                    'method_names': prop_names,
                    'method_fn': callee,
                    #'runtime_signature': runtime_sig,
                })

    print(f" |-> Found {len(matched_entries)} matching global runtime definitions:")

    # Pass 1: Extract signatures and classify by method definitions
    for entry in matched_entries:
        target_node, obj_node = find_method_returned_object(entry['method_fn'])

        signatures = {}
        for method_name in entry['method_names']:
            fn_node = get_method_node_for_property(
                entry['method_fn'], target_node, obj_node, method_name, session
            )
            signatures[method_name] = extract_method_signature(fn_node, session)

        entry['method_signatures'] = signatures
        methods = list(entry['method_signatures'].keys())
        method_count = len(methods)

        # -------------------------------------------------------------------
        # 1-Method Modules
        # -------------------------------------------------------------------
        if method_count == 1:
            method = methods[0]
            sig = entry['method_signatures'][method]
            params = sig['params']
            returns = sig['returns']
            ret = returns[0] if returns else 'None'
            label = 'unknown'

            if params == 0:
                if returns == ['number', 'number']:
                    label = 'env_scoring'
                elif ret == 'None':
                    label = 'trap_call'
                elif ret in ('UnaryExpression', 'FunctionExpression', 'MemberExpression'):
                    label = resolve_cf_env_test(session, entry['global_key'], method)

            elif params in (1, 2):
                if ret == 'FunctionExpression':
                    label = 'string_deobfuscator'
                elif params == 1 and ret == 'ConditionalExpression':
                    label = 'expiration_lock'

            elif params == 3:
                if ret == 'ArrayExpression':
                    label = 'cyclical_array_lookup'
                elif ret == 'BinaryExpression':
                    label = 'murmur3_cf_obfuscation'

            entry['label'] = label
            entry['classification'] = {'label': label, 'method': method}

        # -------------------------------------------------------------------
        # 2-Method Modules
        # -------------------------------------------------------------------
        elif method_count == 2:
            m1, m2 = methods[0], methods[1]
            sig1, sig2 = entry['method_signatures'][m1], entry['method_signatures'][m2]

            label, role_map = classify_dual_methods(m1, sig1, m2, sig2)
            roles = [role_map[m1], role_map[m2]]

            entry['label'] = label
            entry['roles'] = roles
            entry['classification'] = {
                'label': label,
                'roles': roles,
                'role_map': role_map,
            }

    # Pass 2: Log results and bind oracles / fold expressions
    for entry in matched_entries:
        classification = entry['classification']
        key = entry['global_key']
        methods = ", ".join(entry['method_names']) if entry['method_names'] else "<none>"
        label = classification.get('label', 'object_guard')

        detail = ''
        if 'roles' in classification:
            detail = f" {classification['roles']}"
        print(f"   [+] _global[{key!r}] -> Methods: [{methods}]  ==> {label}{detail}")

        try: _key = str(int(float(key))) # FIXME
        except: _key = str(key)

        if label == 'cf_env_test_truthy':
            session.oracle.calls[f'{runtime_object}.{_key}.{classification["method"]}'] = lambda: True
        elif label == 'cf_env_test_falsy':
            session.oracle.calls[f'{runtime_object}.{_key}.{classification["method"]}'] = lambda: False
        elif label == 'expiration_lock':
             session.oracle.calls[f'{runtime_object}.{_key}.{classification["method"]}'] = lambda v: bool(sum(int(c, 16) & 1 for c in v) % 2)
        elif label == 'toggle_gates':
            for i in range(2):
                m_name = entry['method_names'][i]
                if classification['roles'][i] == 'toggle_gate':
                    session.oracle.calls[f'{runtime_object}.{_key}.{m_name}'] = lambda s=[False]: s.insert(0, not s.pop()) or s[0]
                else: # random_gate
                    session.oracle.calls[f'{runtime_object}.{_key}.{m_name}'] = lambda s=['0']: s.insert(0, '1' if s.pop() == '0' else '0') or s[0]
        elif label == 'domain_lock':
            folded_count = 0
            for method_name in entry['method_names']:
                calls, _ = find_method_calls(tree, session, key, method_name)
                for call in calls:
                    parent = call.parent
                    if parent is not None and parent.kind == 'BinaryExpression':
                        op = parent.attr('operator')
                        if op in ('===', '==', '!==', '!='):
                            is_match = (op in ('===', '=='))
                            tree.replace(parent, make('Literal', value=is_match))
                            folded_count += 1
            print(f" |-> Blindly folded {folded_count} domain lock comparison(s)")

    print(f" |-> Defined {len(session.oracle.calls)} oracles")
    #print(session.oracle.calls.keys())
    session.runtime_entries = matched_entries
