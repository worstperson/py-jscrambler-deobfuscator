# ---------------------------------------------------------------------------
# Helpers shared with solveStringArray (cyclical array)
# ---------------------------------------------------------------------------

def is_numeric_literal(node):
    if node.kind != 'Literal':
        return False
    v = node.attr('value')
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _fold_constant_expr(node):
    """Literal-only constant fold: Literal, UnaryExpression, and
    BinaryExpression over already-constant operands. Returns a jsir value
    (see jsir.values) or None if any part of the expression isn't a
    provable constant -- never guesses."""
    if node is None:
        return None
    if node.kind == 'Literal':
        return node.attr('value')
    if node.kind == 'UnaryExpression':
        argument = _fold_constant_expr(node.get('argument'))
        if argument is None:
            return None
        result = V.unary(node.attr('operator'), argument)
        return result if V.is_resolved(result) else None
    if node.kind == 'BinaryExpression':
        left = _fold_constant_expr(node.get('left'))
        right = _fold_constant_expr(node.get('right'))
        if left is None or right is None:
            return None
        result = V.binary(node.attr('operator'), left, right)
        return result if V.is_resolved(result) else None
    return None


def numeric_value(node, session=None, array_state=None):
    """Resolve `node` to an int if it's provably a constant number, else
    None. Handles masked case labels and state updates (`case a ^ b:`,
    `state = x + 1`), not just bare literals.

    Uses a local constant fold rather than session.constant_of: case-test
    expressions are never emitted as evaluated instructions by jsir.flow's
    CFG builder (switch is modeled as control edges only), so CCP never
    resolves them regardless of how simple the expression is. `session` is
    accepted for a possible future extension (e.g. resolving a const
    reference) but isn't used yet.

    `array_state`, when given, is an `_ArrayStateIds` for the state machine
    currently being matched: some obfuscators express every state id --
    case labels, the loop's exit sentinel, the initial value, state
    updates -- as a lookup into a `find_cyclical_array_generators` array
    instead of a bare integer (`case _global[90266].S99HFIc()[59][30][35]:`
    rather than `case 3:`). Those cells aren't meaningful numbers on their
    own, so `array_state` hands out a synthetic int per distinct cell,
    stable for the lifetime of one state machine's match+build, which is
    all that's needed: the CFG only cares that two references to the same
    cell compare equal.
    """
    if is_numeric_literal(node):
        v = node.attr('value')
        return int(v)
    if array_state is not None:
        resolved = array_state.resolve(node)
        if resolved is not None:
            return resolved
    value = _fold_constant_expr(node)
    if value is None or not V.is_resolved(value):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and value == int(value):
        return int(value)
    return None


def is_int_literal(node):
    if node is None or node.kind != 'Literal':
        return False
    v = node.attr('value')
    return isinstance(v, float) and v == int(v)


def _index_chain(node):
    """Walk `node[i][j][k]...` outward-in, returning (base, [i, j, k, ...])
    with indices in left-to-right (outermost-object-first) order, or
    (node, []) if node isn't a computed-integer MemberExpression at all."""
    indices = []
    current = node
    while (current is not None and current.kind == 'MemberExpression' and
           current.attr('computed', False) and is_int_literal(current.get('property'))):
        indices.append(int(current.get('property').attr('value')))
        current = current.get('object')
    indices.reverse()
    return current, indices


# ---------------------------------------------------------------------------
# Cyclical-array boolean comparison fold
# ---------------------------------------------------------------------------

def resolve_boolean_identity(node, generated_arrays):
    """Resolve `node` as `<alias>.<accessor>()[i][j][...]` against a known
    generated array. Returns (True, identity) on a match, even if the
    resolved cell is None, or (False, None) if the shape doesn't match at
    all -- callers must not treat a bare None as a match, since that would
    fold unrelated expressions to `false`.

    `identity` mirrors JS reference equality: two lookups landing on the
    same array cell (a shared list reference, or both unset/None) count as
    equal.
    """
    base, indices = _index_chain(node)
    if not indices:
        return False, None

    if base.kind != 'CallExpression' or base.children_in('arguments'):
        return False, None
    callee = base.get('callee')
    if callee is None or callee.kind != 'MemberExpression' or callee.attr('computed', False):
        return False, None
    name = callee.get('property')
    if name is None or name.kind != 'Identifier':
        return False, None
    alias_key = member_key(callee.get('object'))
    if alias_key is None:
        return False, None

    target = generated_arrays.get((alias_key, name.attr('name')))
    if target is None:
        return False, None

    try:
        for idx in indices:
            target = target[idx]
    except (IndexError, TypeError):
        # Wrong depth / out of range for this array -- not a confident
        # match, so don't pretend we resolved it to an unset cell.
        return False, None

    identity = id(target) if isinstance(target, list) else target
    return True, identity


def member_key(node):
    """A hashable key for a MemberExpression/Identifier chain, used to
    match `<alias>.<prop>` accesses back to the assignment that produced
    them, without going through escodegen. Returns None if any part of the
    chain isn't statically known (a dynamic index, a computed identifier
    key, etc.) -- callers must not treat None as a valid map key."""
    if node is None:
        return None
    if node.kind == 'Identifier':
        return ('id', node.attr('name'))
    if node.kind == 'MemberExpression':
        obj_key = member_key(node.get('object'))
        if obj_key is None:
            return None
        prop = node.get('property')
        if prop is None:
            return None
        if node.attr('computed', False):
            if prop.kind == 'Literal' and is_int_literal(prop):
                return (obj_key, int(prop.attr('value')))
            return None
        if prop.kind == 'Identifier':
            return (obj_key, prop.attr('name'))
        return None
    return None
