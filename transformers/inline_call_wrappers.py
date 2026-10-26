from jsir import Session, walk, node as make, clone
from transformers.helpers.ast_utils import unwrap_expression
from transformers.helpers.globals import is_runtime_object, get_member_key


def is_forwarded_arg(arg, param_names: set[str]) -> bool:
    if not arg:
        return False
    if arg.kind == 'Identifier':
        return arg.attr('name') in param_names or arg.attr('name') == 'arguments'
    if arg.kind == 'MemberExpression':
        obj, prop = arg.get('object'), arg.get('property')
        if obj and obj.kind == 'Identifier' and obj.attr('name') == 'arguments':
            if prop and prop.kind == 'Literal' and isinstance(prop.attr('value'), (int, float)):
                return True
    return False


def is_valid_guard_test(test_node, param_names: set[str], session) -> bool:
    if not test_node:
        return False

    def collect_and_operands(n):
        if n.kind == 'LogicalExpression' and n.attr('operator') == '&&':
            return collect_and_operands(n.get('left')) + collect_and_operands(n.get('right'))
        return [n]

    operands = collect_and_operands(test_node)
    has_runtime = False

    for op in operands:
        if is_runtime_object(op, session):
            has_runtime = True
        elif is_forwarded_arg(op, param_names):
            pass
        else:
            return False

    return has_runtime


def is_guard_return(stmt) -> bool:
    if not stmt:
        return False
    if stmt.kind == 'BlockStatement':
        stmts = stmt.get('body') or []
        if len(stmts) == 0:
            return True
        if len(stmts) == 1:
            stmt = stmts[0]
        else:
            return False

    if stmt.kind != 'ReturnStatement':
        return False
    arg = stmt.get('argument')
    if arg is None:
        return True
    if arg.kind == 'Identifier' and arg.attr('name') == 'undefined':
        return True
    if arg.kind == 'Literal' and arg.attr('value') is None:
        return True
    return False


def is_guard_statement(stmt) -> bool:
    if not stmt or stmt.kind != 'IfStatement':
        return False
    cons = stmt.get('consequent')
    return is_guard_return(cons)


def extract_target_callee(ret_stmt, func_node, param_names: set[str], session):
    if not ret_stmt or ret_stmt.kind != 'ReturnStatement':
        return None
    call = ret_stmt.get('argument')
    if not call or call.kind != 'CallExpression':
        return None

    callee = call.get('callee')
    if not callee or callee.kind != 'MemberExpression':
        return None

    key_obj = callee.get('object')
    if not key_obj or key_obj.kind != 'MemberExpression':
        return None

    if not is_runtime_object(key_obj.get('object'), session):
        return None

    args = call.children_in('arguments')
    if not args:
        return None

    if not all(is_forwarded_arg(a, param_names) for a in args):
        return None

    return callee


def match_wrapper_function(func_node, session):
    if not func_node or func_node.kind not in ('FunctionExpression', 'ArrowFunctionExpression'):
        return None

    body = func_node.get('body')
    if not body:
        return None

    param_names = {
        p.attr('name')
        for p in func_node.children_in('params')
        if p and p.kind == 'Identifier' and p.attr('name')
    }

    if body.kind != 'BlockStatement':
        mock_ret = make('ReturnStatement', argument=body)
        return extract_target_callee(mock_ret, func_node, param_names, session)

    stmts = body.get('body') or []

    if len(stmts) == 1:
        s = stmts[0]
        if s.kind == 'ReturnStatement':
            return extract_target_callee(s, func_node, param_names, session)

        if s.kind == 'IfStatement':
            alt = s.get('alternate')
            if alt is not None and not is_guard_return(alt):
                return None

            test = s.get('test')
            if is_valid_guard_test(test, param_names, session):
                cons = s.get('consequent')
                if cons and cons.kind == 'BlockStatement':
                    c_stmts = cons.get('body') or []
                    cons = c_stmts[0] if len(c_stmts) == 1 else None
                return extract_target_callee(cons, func_node, param_names, session)

        return None

    if len(stmts) == 2:
        guard_stmt, ret_stmt = stmts[0], stmts[1]
        if is_guard_statement(guard_stmt):
            return extract_target_callee(ret_stmt, func_node, param_names, session)
        return None

    return None


def inline_call_wrappers(session: Session) -> None:
    print("--- Applying Transformer: inline_call_wrappers ---")

    runtime_object = getattr(session, 'runtime_object', None)
    if not runtime_object:
        print(" |-> No runtime object.")
        return

    tree = session.tree
    if not hasattr(session, 'wrapper_aliases') or session.wrapper_aliases is None:
        session.wrapper_aliases = {}

    wrapper_map = {}
    for assign in walk(tree.root, kinds={'AssignmentExpression'}):
        if assign.attr('operator') != '=':
            continue
        left = assign.get('left')
        if not left or left.kind != 'MemberExpression':
            continue
        if not is_runtime_object(left.get('object'), session):
            continue

        key = get_member_key(left, as_string=True)
        if key is None:
            continue

        target_callee = match_wrapper_function(assign.get('right'), session)
        if target_callee is not None:
            wrapper_map[key] = (assign, target_callee)
            session.wrapper_aliases[key] = target_callee

    if not wrapper_map:
        print(" |-> No matching call wrapper functions found.")
        return

    print(f" |-> Found {len(wrapper_map)} valid wrapper definitions.")

    for key, (assign_node, _) in wrapper_map.items():
        if not assign_node.live:
            continue
        parent = assign_node.parent
        if parent and parent.kind == 'ExpressionStatement':
            if parent.live:
                tree.remove(parent)
        elif parent and parent.kind == 'SequenceExpression':
            tree.remove(assign_node)
        else:
            tree.replace(assign_node, make('Identifier', name='undefined'))

    replacements = []
    for member in walk(tree.root, kinds={'MemberExpression'}):
        if not member.live:
            continue
        key = get_member_key(member, as_string=True)
        if key not in session.wrapper_aliases:
            continue
        if not is_runtime_object(member.get('object'), session):
            continue
        replacements.append((member, session.wrapper_aliases[key]))

    print(f" |-> Resolving {len(replacements)} wrapper call reference(s)...")
    for member, target_expr in replacements:
        if member.live:
            tree.replace(member, clone(target_expr))
