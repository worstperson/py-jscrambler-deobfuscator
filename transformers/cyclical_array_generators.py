from jsir import Session, walk
from transformers.helpers.common import member_key, is_int_literal

# ---------------------------------------------------------------------------
# Cyclical-array generator: materialise the array
# ---------------------------------------------------------------------------
#
# Straight port of controlFlowUnflattener.generate_cyclical_array /
# _set_index. Pure Python bookkeeping, nothing IR-shaped about it, so it
# doesn't change at all going from paths to jsir nodes.

def _generate_cyclical_array(size, offset_value, key):
    matrix = [[] for _ in range(size)]
    for current_row in range(size):
        for source_row in range(size - 1, -1, -1):
            base_offset = key[0]
            if base_offset == 0:
                continue
            modulus = base_offset
            dest_col = (source_row + offset_value * current_row) % modulus
            if 0 <= source_row < len(matrix):
                _set_index(matrix[current_row], dest_col, matrix[source_row])
    return matrix


def _set_index(lst, index, value):
    if index >= len(lst):
        lst.extend([None] * (index - len(lst) + 1))
    lst[index] = value


# ---------------------------------------------------------------------------
# Cyclical-array generator: detection
# ---------------------------------------------------------------------------


def find_cyclical_array_generators(session: Session) -> None:
    """Every `<alias> = <call>(<size>, <offset>, [<key>])` assignment,
    where alias is later invoked as `<alias>.<name>()` somewhere in the
    tree. Returns {(alias_key, name): materialised_matrix}.

    Matches purely on the call's argument shape, not on what the callee
    does internally -- the callee can be an empty function, an IIFE
    returning an object, or anything else. The one thing that has to
    exist for the array to be useful is a later zero-arg method call on
    the alias; that call's property name is the array's accessor name.
    """
    print("--- Applying Transformer: cyclical_array_generators ---")
    generated = {}
    aliases = {}  # alias_key -> (matrix, assign_node)

    for assign in walk(session.tree.root, kinds=('AssignmentExpression',)):
        left = assign.get('left')
        right = assign.get('right')
        if left is None or right is None or right.kind != 'CallExpression':
            continue

        args = right.children_in('arguments')
        if len(args) != 3:
            continue
        size_arg, offset_arg, key_arg = args
        if not (is_int_literal(size_arg) and is_int_literal(offset_arg)):
            continue
        if key_arg.kind != 'ArrayExpression':
            continue
        key_elements = key_arg.children_in('elements')
        if len(key_elements) != 1 or not is_int_literal(key_elements[0]):
            continue

        alias_key = member_key(left)
        if alias_key is None or alias_key in aliases:
            continue

        size = int(size_arg.attr('value'))
        offset = int(offset_arg.attr('value'))
        key = [int(key_elements[0].attr('value'))]
        matrix = _generate_cyclical_array(size, offset, key)
        aliases[alias_key] = matrix
        print(f" |-> Found cyclical array generator: size={size}, offset={offset}, key={key}")

    if aliases:
        for call in walk(session.tree.root, kinds=('CallExpression',)):
            callee = call.get('callee')
            if callee is None or callee.kind != 'MemberExpression' or callee.attr('computed', False):
                continue
            if call.children_in('arguments'):
                continue
            name = callee.get('property')
            if name is None or name.kind != 'Identifier':
                continue
            alias_key = member_key(callee.get('object'))
            if alias_key not in aliases:
                continue
            full_key = (alias_key, name.attr('name'))
            if full_key not in generated:
                generated[full_key] = aliases[alias_key]
                print(f"   |-> Accessor '{name.attr('name')}' resolves to generated array")

    session.generated_arrays = generated
