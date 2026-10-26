"""
solve_state_machine -- resolve calls into an opcode-dispatch closure.
"""

from jsir import Session, walk, clone, node as make, values as V
from transformers.helpers.literals import value_to_node as _value_to_node
from transformers.helpers.common import member_key as _member_key


def _is_int_literal(node):
    if node is None or node.kind != 'Literal':
        return False
    v = node.attr('value')
    return isinstance(v, float) and v == int(v)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class StateMachine:
    __slots__ = ('object_key', 'setter_name', 'calculator_name', 'logic_map')

    def __init__(self, object_key, setter_name, calculator_name, logic_map):
        self.object_key = object_key
        self.setter_name = setter_name
        self.calculator_name = calculator_name
        self.logic_map = logic_map


def find_state_machines(session):
    """Every `<obj> = (function(state){ return {setter, calculator} })()`
    assignment in the tree, in document order."""
    machines = []
    for assign in walk(session.tree.root, kinds=('AssignmentExpression',)):
        machine = _match_state_machine_assignment(assign)
        if machine is not None:
            machines.append(machine)
    return machines


def _match_state_machine_assignment(assign):
    right = assign.get('right')
    if right is None or right.kind != 'CallExpression':
        return None
    callee = right.get('callee')
    if callee is None or callee.kind != 'FunctionExpression':
        return None

    body_stmts = _block_body(callee.get('body'))
    if len(body_stmts) != 1 or body_stmts[0].kind != 'ReturnStatement':
        return None
    return_arg = body_stmts[0].get('argument')
    if return_arg is None or return_arg.kind != 'ObjectExpression':
        return None
    properties = return_arg.children_in('properties')
    if len(properties) != 2:
        return None

    params = callee.children_in('params')
    if len(params) != 1 or params[0].kind != 'Identifier':
        return None
    state_var_name = params[0].attr('name')

    print(f" |-> Found potential State Machine pattern with state var '{state_var_name}'")

    setter_name = None
    calculator_name = None
    logic_map = {}

    for prop in properties:
        if prop.kind != 'Property':
            continue
        prop_value = prop.get('value')
        if prop_value is None or prop_value.kind != 'FunctionExpression':
            continue
        prop_name = _property_name(prop)
        if prop_name is None:
            continue

        print(f"    |-> Analyzing property: '{prop_name}'")
        prop_body = _block_body(prop_value.get('body'))

        if _is_setter(prop_value, prop_body, state_var_name):
            setter_name = prop_name
            print("      |-> Identified as SETTER.")
            continue

        switch_stmt = next((s for s in prop_body if s.kind == 'SwitchStatement'), None)
        if switch_stmt is not None:
            disc = switch_stmt.get('discriminant')
            if disc is not None and disc.kind == 'Identifier' and disc.attr('name') == state_var_name:
                calculator_name = prop_name
                print("      |-> Identified as CALCULATOR. Parsing cases...")
                logic_map = _parse_calculator_cases(switch_stmt)

    if not (setter_name and calculator_name and logic_map):
        print(" |-> [FAIL] Could not identify both a setter and a calculator from the properties.")
        return None

    object_key = _member_key(assign.get('left'))
    if object_key is None:
        print(" |-> [FAIL] State machine's own assignment target isn't a statically-keyed identifier/member chain.")
        return None

    print(" |-> State Machine fully parsed!")
    print(f"   |-> Setter Fn:   '{setter_name}'")
    print(f"   |-> Calculator Fn: '{calculator_name}'")
    print(f"   |-> Logic map size: {len(logic_map)}")

    return StateMachine(object_key, setter_name, calculator_name, logic_map)


def _block_body(block):
    if block is None or block.kind != 'BlockStatement':
        return []
    return list(block.children_in('body'))


def _property_name(prop):
    key = prop.get('key')
    if key is None:
        return None
    if key.kind == 'Identifier':
        return key.attr('name')
    if key.kind == 'Literal':
        return key.attr('value')
    return None


def _is_setter(prop_value, prop_body, state_var_name):
    params = prop_value.children_in('params')
    if len(params) != 1:
        return False
    for stmt in prop_body:
        if stmt.kind != 'ExpressionStatement':
            continue
        expr = stmt.get('expression')
        if expr is None or expr.kind != 'AssignmentExpression':
            continue
        left = expr.get('left')
        if left is not None and left.kind == 'Identifier' and left.attr('name') == state_var_name:
            return True
    return False


def _parse_calculator_cases(switch_stmt):
    logic_map = {}
    for case in switch_stmt.children_in('cases'):
        assignment = None
        for stmt in case.children_in('consequent'):
            if stmt.kind != 'ExpressionStatement':
                continue
            expr = stmt.get('expression')
            if expr is not None and expr.kind == 'AssignmentExpression':
                assignment = expr
                break
        test_node = case.get('test')
        if assignment is not None and test_node is not None and test_node.kind == 'Literal':
            case_val = test_node.attr('value')
            logic_map[case_val] = assignment.get('right')
            #print(f"        |-> Stored logic for case {case_val}.")
    print(f"        |-> Stored logic for {len(logic_map)} cases.")
    return logic_map


# ---------------------------------------------------------------------------
# Logic-node evaluator (case body arithmetic, argument-array aware)
# ---------------------------------------------------------------------------

def _is_numeric_str(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _reconstruct_logic(node, arg_nodes):
    """
    Substitutes arg_nodes into logic_node. If all components are literals,
    folds to a concrete value. Otherwise, reconstructs the AST expression.
    """
    if node is None:
        raise ValueError('Received None logic node.')

    # Argument reference: e.g. v1[0], args[1]
    if node.kind == 'MemberExpression':
        prop = node.get('property')
        if prop is not None and prop.kind == 'Literal' and isinstance(prop.attr('value'), (int, float)):
            idx = int(prop.attr('value'))
            if 0 <= idx < len(arg_nodes):
                return clone(arg_nodes[idx])
        raise ValueError(f'Unsupported member expression in calculator logic: {node}')

    if node.kind == 'Literal':
        return clone(node)

    if node.kind == 'Identifier' and node.attr('name') == 'undefined':
        return make('Identifier', name='undefined')

    if node.kind == 'UnaryExpression':
        arg = _reconstruct_logic(node.get('argument'), arg_nodes)
        op = node.attr('operator')
        if arg.kind == 'Literal' and isinstance(arg.attr('value'), (int, float)):
            res = V.unary(op, float(arg.attr('value')))
            if V.is_resolved(res):
                return _value_to_node(res)
        return make('UnaryExpression', operator=op, argument=arg, prefix=True)

    if node.kind == 'BinaryExpression':
        left = _reconstruct_logic(node.get('left'), arg_nodes)
        right = _reconstruct_logic(node.get('right'), arg_nodes)
        op = node.attr('operator')

        # If both sides are literals, attempt constant folding
        if left.kind == 'Literal' and right.kind == 'Literal':
            l_val = left.attr('value')
            r_val = right.attr('value')
            l_num = float(l_val) if isinstance(l_val, (int, float)) or (isinstance(l_val, str) and _is_numeric_str(l_val)) else None
            r_num = float(r_val) if isinstance(r_val, (int, float)) or (isinstance(r_val, str) and _is_numeric_str(r_val)) else None
            if l_num is not None and r_num is not None:
                res = V.binary(op, l_num, r_num)
                if V.is_resolved(res):
                    return _value_to_node(res)

        return make('BinaryExpression', operator=op, left=left, right=right)

    raise ValueError(f'Unsupported node kind in calculator logic: {node.kind}')


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def _match_member_call(callee_node):
    """For a CallExpression's callee, return (object_key, prop_name) if
    it's `<obj>.<prop>(...)`/`<obj>[<prop>](...)`, else None."""
    if callee_node is None or callee_node.kind != 'MemberExpression':
        return None
    obj_key = _member_key(callee_node.get('object'))
    if obj_key is None:
        return None
    prop = callee_node.get('property')
    if prop is None:
        return None
    if callee_node.attr('computed', False):
        prop_name = prop.attr('value') if prop.kind == 'Literal' else None
    else:
        prop_name = prop.attr('name') if prop.kind == 'Identifier' else None
    if prop_name is None:
        return None
    return obj_key, prop_name


def solve_state_machine(session: Session) -> None:
    print("--- Applying Transformer: solve_state_machine ---")

    tree = session.tree
    machines = find_state_machines(session)
    if not machines:
        return

    pending_statement_removals = []
    current_state = {id(m): None for m in machines}

    for call in walk(tree.root, kinds=('CallExpression',)):
        if not tree.contains(call):
            continue  # consumed by an earlier replacement in this same walk

        callee = call.get('callee')
        match = _match_member_call(callee)
        if match is None:
            continue
        obj_key, prop_name = match

        machine = next((m for m in machines if m.object_key == obj_key), None)
        if machine is None:
            continue

        if prop_name == machine.setter_name:
            _resolve_setter_call(tree, call, machine, current_state, pending_statement_removals)
        elif prop_name == machine.calculator_name:
            _resolve_calculator_call(session, call, machine, current_state)

    for stmt in pending_statement_removals:
        if stmt.live:
            tree.remove(stmt)


def _resolve_setter_call(tree, call, machine, current_state, pending_statement_removals):
    args = call.children_in('arguments')
    if not args:
        return
    res = _static_eval(args[0])
    if res is _UNRESOLVED:
        print(f' |-> Could not determine state value for setter call (nid={call.nid})')
        return

    current_state[id(machine)] = res
    parent = call.parent
    if parent is not None and parent.kind == 'ExpressionStatement':
        pending_statement_removals.append(parent)
    else:
        tree.replace(call, make('Identifier', name='undefined'))


def _resolve_calculator_call(session, call, machine, current_state):
    tree = session.tree
    args = list(call.children_in('arguments'))

    # Handle nested setter calls in arguments (e.g. calc('21', 0, setter(0)))
    setter_arg_indices = set()
    for i, arg_node in enumerate(args):
        if arg_node.kind != 'CallExpression':
            continue
        inner_match = _match_member_call(arg_node.get('callee'))
        if inner_match is None:
            continue
        inner_obj_key, inner_prop_name = inner_match
        if inner_obj_key != machine.object_key or inner_prop_name != machine.setter_name:
            continue

        inner_args = arg_node.children_in('arguments')
        if not inner_args:
            print(f' |-> Nested setter call takes no argument, cannot resolve state (nid={call.nid})')
            return

        inner_res = _static_eval(inner_args[0])
        if inner_res is _UNRESOLVED:
            print(f' |-> Could not resolve nested setter argument for call (nid={call.nid})')
            return

        current_state[id(machine)] = inner_res
        setter_arg_indices.add(i)
        tree.replace(arg_node, make('Identifier', name='undefined'))

    state = current_state[id(machine)]
    if state is None:
        print(f' |-> Calculator called before state was set (nid={call.nid})')
        return

    logic_node = machine.logic_map.get(state)
    if logic_node is None:
        print(f' |-> No logic for state {state!r} in call (nid={call.nid})')
        return

    # Prepare argument nodes (pass AST nodes directly, replacing nested setters with undefined)
    arg_nodes = []
    for i, arg_node in enumerate(args):
        if i in setter_arg_indices:
            arg_nodes.append(make('Identifier', name='undefined'))
        else:
            arg_nodes.append(arg_node)

    try:
        replacement = _reconstruct_logic(logic_node, arg_nodes)
    except ValueError as e:
        print(f' |-> Failed evaluation for call (nid={call.nid}): {e}')
        return

    tree.replace(call, replacement)

_UNRESOLVED = object()

def _static_eval(node):
    """Best-effort constant evaluation of a call argument: Literal,
    negative-number UnaryExpression, and `undefined`.
    """
    if node is None:
        return _UNRESOLVED
    if node.kind == 'Literal':
        return node.attr('value')
    if node.kind == 'Identifier' and node.attr('name') == 'undefined':
        return V.UNDEF
    if node.kind == 'UnaryExpression' and node.attr('operator') == '-':
        inner = _static_eval(node.get('argument'))
        if inner is _UNRESOLVED or isinstance(inner, bool) or not isinstance(inner, (int, float)):
            return _UNRESOLVED
        return -inner
    return _UNRESOLVED
