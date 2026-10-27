from jsir import Session, walk, node as make, clone
from transformers.helpers.ast_utils import unwrap_expression
from transformers.helpers.globals import (
    is_runtime_object,
    get_member_key,
    resolve_identifier,
)


def is_equivalent(n1, n2, scopes=None, visited=None) -> bool:
    if n1 is n2:
        return True
    if not (n1 and n2) or n1.kind != n2.kind:
        return False

    if visited is None:
        visited = set()

    pair = (id(n1), id(n2))
    if pair in visited:
        return True
    visited.add(pair)

    if n1.kind == 'Identifier':
        if n1.attr('name') == n2.attr('name'):
            return True
        if not scopes:
            return False
        r1 = resolve_identifier(n1, scopes)
        r2 = resolve_identifier(n2, scopes)
        if r1 is n1 and r2 is n2:
            return False
        return is_equivalent(r1, r2, scopes, visited)

    if n1.kind == 'Literal':
        return str(n1.attr('value')) == str(n2.attr('value'))

    if n1.kind == 'MemberExpression':
        if n1.attr('computed') != n2.attr('computed'):
            return False
        return (
            is_equivalent(n1.get('object'), n2.get('object'), scopes, visited) and
            is_equivalent(n1.get('property'), n2.get('property'), scopes, visited)
        )

    return False


def is_wrapper_function(node, scopes=None):
    if not node or node.kind != 'FunctionExpression':
        return None

    body = node.get('body')
    stmts = (body.get('body') or []) if body else []
    if len(stmts) != 1 or stmts[0].kind != 'ReturnStatement':
        return None

    ret = stmts[0].get('argument')
    if not ret or ret.kind != 'ConditionalExpression':
        return None

    test, cons, alt = ret.get('test'), ret.get('consequent'), ret.get('alternate')
    if not test or test.kind != 'BinaryExpression':
        return None

    op = test.attr('operator')
    if op not in ('===', '==', '!==', '!='):
        return None

    left, right = test.get('left'), test.get('right')
    typeof_expr = None
    if left and left.kind == 'UnaryExpression' and left.attr('operator') == 'typeof' and right and right.kind == 'Literal' and right.attr('value') == 'function':
        typeof_expr = left
    elif right and right.kind == 'UnaryExpression' and right.attr('operator') == 'typeof' and left and left.kind == 'Literal' and left.attr('value') == 'function':
        typeof_expr = right

    if not typeof_expr or getattr(typeof_expr.get('argument'), 'kind', None) != 'MemberExpression':
        return None

    target = typeof_expr.get('argument')

    if op in ('!==', '!='):
        cons, alt = alt, cons

    if not is_equivalent(alt, target, scopes):
        return None

    if not (cons and cons.kind == 'CallExpression'):
        return None
    callee = unwrap_expression(cons.get('callee'))
    if not (callee and callee.kind == 'MemberExpression' and get_member_key(callee) == 'apply'):
        return None
    if not is_equivalent(callee.get('object'), target, scopes):
        return None

    args = cons.get('arguments') or []
    if len(args) != 2 or not is_equivalent(args[0], target.get('object'), scopes):
        return None
    if getattr(args[1], 'kind', None) != 'Identifier' or args[1].attr('name') != 'arguments':
        return None

    return target


def get_base_object(node):
    curr = node
    while curr:
        if curr.kind == 'AssignmentExpression':
            curr = curr.get('right')
        elif curr.kind == 'MemberExpression':
            curr = curr.get('object')
        elif curr.kind == 'ParenthesizedExpression':
            curr = curr.get('expression')
        else:
            break
    return curr


def inline_wrapper_functions(session: Session) -> None:
    print("--- Applying Transformer: inline_wrapper_functions ---")

    session.wrapper_aliases = {}
    runtime_object = getattr(session, 'runtime_object', None)
    if not runtime_object:
        print(" |-> No runtime base object found.")
        return

    tree = session.tree

    wrapper_map = {}
    for assign in walk(tree.root, kinds={'AssignmentExpression'}):
        if assign.attr('operator') != '=':
            continue
        left = assign.get('left')
        if not left or left.kind != 'MemberExpression':
            continue

        base_obj = get_base_object(left.get('object'))
        if not is_runtime_object(base_obj, session):
            continue

        target_expr = is_wrapper_function(assign.get('right'), session.scopes)
        if target_expr:
            key = get_member_key(left, as_string=True)
            if key is not None:
                session.wrapper_aliases[key] = target_expr
                wrapper_map[key] = assign

    if not wrapper_map:
        print(" |-> No wrapper function definitions found.")
        return

    print(f" |-> Found {len(wrapper_map)} wrapper definitions.")
    print(" |-> Cleaning up wrapper assignment statements...")
    for key, assign_node in wrapper_map.items():
        if not getattr(assign_node, 'is_attached', True):
            continue
        parent = assign_node.parent
        if parent and parent.kind == 'ExpressionStatement':
            tree.remove(parent)
        elif parent and parent.kind == 'SequenceExpression':
            tree.remove(assign_node)
        else:
            tree.replace(assign_node, make('Identifier', name='undefined'))


def resolve_wrapper_aliases(session: Session) -> None:
    print("--- Applying Transformer: resolve_wrapper_aliases ---")

    aliases = getattr(session, 'wrapper_aliases', None)
    if not aliases:
        return
    runtime_object = getattr(session, 'runtime_object', None)
    if not runtime_object:
        return

    tree = session.tree
    replacements = []
    for member in walk(tree.root, kinds={'MemberExpression'}):
        key = get_member_key(member, as_string=True)
        if key not in aliases:
            continue
        if not is_runtime_object(member.get('object'), session):
            continue
        replacements.append((member, aliases[key]))

    print(f" |-> Resolving {len(replacements)} wrapper alias reference(s)...")
    for member, target_expr in replacements:
        if not member.live:
            continue
        tree.replace(member, clone(target_expr))
