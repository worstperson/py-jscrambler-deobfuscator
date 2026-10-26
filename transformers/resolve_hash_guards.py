"""
resolve_hash_guards — fold and unroll murmur3-hash control-flow guards.

This obfuscator wraps `if`/`for`/`while` conditions in a call to a bundled
murmur3-32 hash (assigned onto the runtime object as `_global[key].name`),
compared against a hardcoded target:

    if (_global[232388].c0nSvxa((2).toString(), (2).toString().length, 43911)
            !== 928544261) { ... }

Every argument this obfuscator ever passes is one of: a numeric literal, a
`.toString()` of one, a `.length` of that string, or a variable CCP can
already resolve to a concrete number. The hash isn't a secret, it's
arithmetic -- every one of these conditions is always-true or always-false,
never actually data-dependent at runtime.

Two shapes show up:

  * An *invariant* guard: every argument is already constant, so its truth
    value doesn't depend on anything the program computes. `if`/bare-`for`
    forms of this collapse in place.

  * A *counter-driven* loop: `for (var C = INIT; guard(C, ...) CMP TARGET;
    C += STEP) { BODY }`, or the equivalent with the init/increment split
    across an ordinary `while` (init as the preceding statement, increment
    as the loop body's last statement) -- and BODY never reads C. This is
    CCP's actual blind spot: only concrete execution finds the trip count.
    We simulate C forward through the same hash, then replace the loop with
    that many copies of BODY. Whatever BODY's own arithmetic produces (this
    obfuscator's trailing "sanity check" `if`, re-testing the loop's exit
    counter against a second target) then folds on the *next* fold_constants
    round via ordinary CCP -- this module doesn't need to know that
    connection exists, only that unrolling turns a hash-guarded loop into
    plain sequential statements CCP already knows how to fold.

Deliberately doesn't touch CCP/Oracle: `.toString()`/`.length` on a concrete
value aren't something the dense solver's call-resolution models (see
_callee_path in ccp.py -- it resolves dotted *global* paths, not built-in
prototype methods on primitives), so guard evaluation here is a small,
self-contained interpreter over exactly the argument shapes this obfuscator
emits, not a general one.

Schedule this after fold_constants, and let a normal fold_constants /
run_until_stable round follow it -- the unrolled arithmetic and the
trailing sanity-check `if` both need one more ordinary CCP pass to actually
fold, not anything this module does itself.
"""

from collections import namedtuple, Counter

from jsir import Session, walk, node as make, clone
from jsir.values import number_to_string
from transformers.helpers.murmur3 import murmur3_x86_32_bytes
from transformers.helpers.ast_utils import (
    splice_statements,
    unwrap_expression,
    references_name,
    is_iife_call,
    read_member_key_and_type,
)


_MAX_UNROLL = 20


# ---------------------------------------------------------------------------
# The hash itself
# ---------------------------------------------------------------------------

def murmur3_guard(v1, h2, o1):
    bytes_data = [ord(c) & 0xFF for c in v1[:h2]]
    # Pad with 0s if h2 exceeds len(v1) to retain original _char_code safety behavior
    if len(bytes_data) < h2:
        bytes_data.extend([0] * (h2 - len(bytes_data)))

    return murmur3_x86_32_bytes(
        bytes_data, length=h2, seed=o1 & 0xFFFFFFFF, mix_block=True
    )


# ---------------------------------------------------------------------------
# Finding the hash provider (shape-based, like remove_code_locks)
# ---------------------------------------------------------------------------

HashProvider = namedtuple('HashProvider', 'base_name base_key has_name method_name')


def _read_key(member_node):
    prop = member_node.get('property')
    if prop is None:
        return None, False
    if prop.kind == 'Literal':
        val = prop.attr('value')
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val), False
    if prop.kind == 'Identifier' and not member_node.attr('computed', False):
        return prop.attr('name'), True
    return None, False


def find_hash_provider(tree, base_name):
    for assign in walk(tree.root, kinds={'AssignmentExpression'}):
        left, right = assign.get('left'), assign.get('right')
        if left is None or right is None or left.kind != 'MemberExpression':
            continue
        obj = left.get('object')
        if obj is None or obj.kind != 'Identifier' or obj.attr('name') != base_name:
            continue
        key, has_name = read_member_key_and_type(left)
        if key is None:
            continue
        if not is_iife_call(right):
            continue
        method_name = _find_hash_method(right)
        if method_name is not None:
            return HashProvider(base_name, key, has_name, method_name)
    return None


def _find_hash_method(call_node):
    """A 3-parameter function inside this IIFE whose body has both a loop
    and a fallthrough switch on the tail-byte count -- the shape this hash
    keeps regardless of what names get shuffled around it."""
    for fn in walk(call_node, kinds={'FunctionExpression', 'ArrowFunctionExpression'}):
        if len(fn.children_in('params')) != 3:
            continue
        body = fn.get('body')
        if body is None:
            continue
        has_loop = any(True for _ in walk(body, kinds={'ForStatement', 'WhileStatement'}))
        has_switch = any(True for _ in walk(body, kinds={'SwitchStatement'}))
        if not (has_loop and has_switch):
            continue
        name = _returned_property_name(call_node, fn)
        if name is not None:
            return name
    return None


def _returned_property_name(call_node, fn_node):
    target_name = None
    if fn_node.parent is not None and fn_node.parent.kind == 'VariableDeclarator' \
            and fn_node.parent.get('init') is fn_node:
        id_node = fn_node.parent.get('id')
        if id_node is not None and id_node.kind == 'Identifier':
            target_name = id_node.attr('name')

    for ret in walk(call_node, kinds={'ReturnStatement'}):
        arg = ret.get('argument')
        if arg is None or arg.kind != 'ObjectExpression':
            continue
        for prop in arg.children_in('properties'):
            if prop.kind != 'Property':
                continue
            key, value = prop.get('key'), prop.get('value')
            if key is None or key.kind != 'Identifier' or value is None:
                continue
            if value is fn_node:
                return key.attr('name')
            if target_name is not None and value.kind == 'Identifier' and \
                    value.attr('name') == target_name:
                return key.attr('name')
    return None


# ---------------------------------------------------------------------------
# Evaluating one guard
# ---------------------------------------------------------------------------

def _resolve_number(node, session, override=None):
    if node is None:
        return None

    # 1. Fast path: check Literal directly (never touches scopes or facts)
    if node.kind == 'Literal':
        val = node.attr('value')
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
        return None

    # 2. Fast path: check override by identifier name (never touches scopes)
    if node.kind == 'Identifier' and override is not None:
        name = node.attr('name')
        if name in override:
            return float(override[name])

    # 3. Unary negation
    if node.kind == 'UnaryExpression' and node.attr('operator') == '-':
        val = _resolve_number(node.get('argument'), session, override)
        return -val if val is not None else None

    # 4. Slow path: scope binding lookup
    if node.kind == 'Identifier' and override is not None:
        binding = session.scopes.binding_of(node)
        if binding is not None and binding in override:
            return float(override[binding])

    # 5. Slow path: CCP solver constant lookup
    value = session.constant_of(node)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _resolve_string(node, session, override=None):
    """The two string-producing shapes this obfuscator ever passes as v1: a
    numeric-literal `.toString()`, or a resolvable variable `.toString()`'d."""
    if node.kind != 'CallExpression':
        return None
    callee = node.get('callee')
    if callee is None or callee.kind != 'MemberExpression' or callee.attr('computed', False):
        return None
    prop = callee.get('property')
    if prop is None or prop.kind != 'Identifier' or prop.attr('name') != 'toString':
        return None
    if callee.children_in('arguments'):
        return None
    number = _resolve_number(unwrap_expression(callee.get('object')), session, override)
    return None if number is None else number_to_string(number)


def _resolve_length(node, session, v1_string, override=None):
    """`X.length` where X is the same `.toString()` call as v1 -- matched by
    shape, not identity, since the obfuscator repeats the call rather than
    reusing a variable."""
    if node.kind != 'MemberExpression' or node.attr('computed', False):
        return None
    prop = node.get('property')
    if prop is None or prop.kind != 'Identifier' or prop.attr('name') != 'length':
        return None
    if _resolve_string(node.get('object'), session, override) is None:
        return None
    return float(len(v1_string))


def _match_guard_call(node, provider):
    """The three argument nodes of a call to the discovered hash function,
    or None."""
    if node.kind != 'CallExpression':
        return None
    callee = node.get('callee')
    if callee is None or callee.kind != 'MemberExpression' or callee.attr('computed', False):
        return None
    prop = callee.get('property')
    if prop is None or prop.kind != 'Identifier' or prop.attr('name') != provider.method_name:
        return None
    obj = callee.get('object')
    if obj is None or obj.kind != 'MemberExpression':
        return None
    key, has_name = read_member_key_and_type(obj)
    if key != provider.base_key or has_name != provider.has_name:
        return None
    inner = obj.get('object')
    if inner is None or inner.kind != 'Identifier' or inner.attr('name') != provider.base_name:
        return None
    args = node.children_in('arguments')
    return args if len(args) == 3 else None


_GuardStats = Counter   # keys: 'seen', 'call_shape_mismatch', 'v1_unresolved',
                         # 'h2_unresolved', 'o1_unresolved', 'target_unresolved', 'folded'

def evaluate_guard(test, provider, session, override=None, stats=None):
    if stats is not None:
        stats['seen'] += 1
    if test.kind != 'BinaryExpression' or test.attr('operator') not in ('!==', '==='):
        return None
    left, right = test.get('left'), test.get('right')

    args = _match_guard_call(left, provider)
    target = right
    if args is None:
        args = _match_guard_call(right, provider)
        target = left
    if args is None:
        if stats is not None:
            stats['call_shape_mismatch'] += 1
        return None

    target_value = _resolve_number(target, session, override)
    if target_value is None:
        if stats is not None:
            stats['target_unresolved'] += 1
        return None

    v1 = _resolve_string(args[0], session, override)
    if v1 is None:
        if stats is not None:
            stats['v1_unresolved'] += 1
        return None
    h2 = _resolve_length(args[1], session, v1, override)
    if h2 is None:
        if stats is not None:
            stats['h2_unresolved'] += 1
        return None
    o1 = _resolve_number(args[2], session, override)
    if o1 is None:
        if stats is not None:
            stats['o1_unresolved'] += 1
        return None

    if stats is not None:
        stats['folded'] += 1
    equal = float(murmur3_guard(v1, int(h2), int(o1))) == target_value
    return equal if test.attr('operator') == '===' else not equal


def _read_step(update, name):
    if update.kind == 'UpdateExpression':
        arg = update.get('argument')
        if arg is None or arg.kind != 'Identifier' or arg.attr('name') != name:
            return None
        return 1.0 if update.attr('operator') == '++' else -1.0
    if update.kind == 'AssignmentExpression' and update.attr('operator') in ('+=', '-='):
        left, right = update.get('left'), update.get('right')
        if left is None or left.kind != 'Identifier' or left.attr('name') != name:
            return None
        if right is None or right.kind != 'Literal' or not isinstance(right.attr('value'), float):
            return None
        magnitude = right.attr('value')
        return magnitude if update.attr('operator') == '+=' else -magnitude
    return None


def _simulate_trip_count(test, provider, session, counter_name, start, step):
    counter = start
    for iterations in range(_MAX_UNROLL):
        truth = evaluate_guard(test, provider, session, {counter_name: counter})
        if truth is None:
            return None
        if not truth:
            return iterations
        counter += step
    return None


# ---------------------------------------------------------------------------
# Pass 1: counter-driven loops (for-shaped and while-shaped)
# ---------------------------------------------------------------------------

def _analyze_for_loop(node, provider, session):
    init, test, update, body = (node.get('init'), node.get('test'),
                                 node.get('update'), node.get('body'))
    if init is None or test is None or update is None or body is None:
        return None

    # Handle both `var x = 1` and `x = 1`
    if init.kind == 'VariableDeclaration':
        decls = init.children_in('declarations')
        if len(decls) != 1:
            return None
        counter_id, counter_init = decls[0].get('id'), decls[0].get('init')
    elif init.kind == 'AssignmentExpression' and init.attr('operator', '=') == '=':
        counter_id, counter_init = init.get('left'), init.get('right')
    else:
        return None

    if counter_id is None or counter_id.kind != 'Identifier' or counter_init is None:
        return None

    start = _resolve_number(counter_init, session)
    if start is None:
        return None
    name = counter_id.attr('name')
    step = _read_step(update, name)
    if step is None or not references_name(test, name):
        return None

    body_statements = list(body.children_in('body')) if body.kind == 'BlockStatement' else [body]
    if any(references_name(stmt, name) for stmt in body_statements):
        return None

    # Key simulation by identifier name so it works regardless of scope binding
    trip_count = _simulate_trip_count(test, provider, session, name, start, step)
    if trip_count is None:
        return None

    return {'body': body_statements, 'trip_count': trip_count, 'init_stmt': None}

def _extract_counter_name(test, session):
    if test.kind != 'BinaryExpression':
        return None
    names = {n.attr('name') for n in walk(test, kinds={'Identifier'})
             if session.scopes.binding_of(n) is not None}
    return names.pop() if len(names) == 1 else None

def _analyze_while_loop(node, provider, session):
    test, body = node.get('test'), node.get('body')
    if test is None or body is None or body.kind != 'BlockStatement':
        return None

    body_statements = list(body.children_in('body'))
    if not body_statements or body_statements[-1].kind != 'ExpressionStatement':
        return None

    # 1. Extract the counter name directly from the trailing increment statement
    increment_expr = body_statements[-1].get('expression')
    if increment_expr is None:
        return None
    if increment_expr.kind == 'UpdateExpression':
        counter_id = increment_expr.get('argument')
    elif increment_expr.kind == 'AssignmentExpression':
        counter_id = increment_expr.get('left')
    else:
        return None

    if counter_id is None or counter_id.kind != 'Identifier':
        return None
    name = counter_id.attr('name')

    # 2. Verify step and references
    step = _read_step(increment_expr, name)
    if step is None or not references_name(test, name):
        return None

    real_body = body_statements[:-1]
    if any(references_name(stmt, name) for stmt in real_body):
        return None

    # 3. Match the preceding init statement
    index = node.index_in_parent()
    if index is None or index == 0:
        return None
    siblings = node.parent.children.get(node.parent_slot)
    prev = siblings[index - 1]
    if prev.kind != 'ExpressionStatement':
        return None
    prev_expr = prev.get('expression')
    if prev_expr is None or prev_expr.kind != 'AssignmentExpression' or \
            prev_expr.attr('operator', '=') != '=':
        return None
    prev_left = prev_expr.get('left')
    if prev_left is None or prev_left.kind != 'Identifier' or prev_left.attr('name') != name:
        return None
    start = _resolve_number(prev_expr.get('right'), session)
    if start is None:
        return None

    # 4. Simulate using the identifier name (matching _analyze_for_loop)
    trip_count = _simulate_trip_count(test, provider, session, name, start, step)
    if trip_count is None:
        return None

    return {'body': real_body, 'trip_count': trip_count, 'init_stmt': prev}


def unroll_counted_loops(session: Session, provider: HashProvider) -> int:
    tree = session.tree

    # PHASE 1: Read-only Analysis (0 cache invalidations)
    plans = []

    for node in walk(tree.root, kinds={'ForStatement'}):
        plan = _analyze_for_loop(node, provider, session)
        if plan is not None:
            plans.append((node, plan, None))

    for node in walk(tree.root, kinds={'WhileStatement'}):
        plan = _analyze_while_loop(node, provider, session)
        if plan is not None:
            plans.append((node, plan, plan.get('init_stmt')))

    # PHASE 2: Mutation (Apply all splices at once)
    count = 0
    for node, plan, init_stmt in plans:
        if not node.live:
            continue
        if init_stmt is not None and init_stmt.live:
            tree.remove(init_stmt)
        _apply_unroll(tree, node, plan)
        count += 1

    return count


def _apply_unroll(tree, node, plan):
    unrolled = []
    for _ in range(plan['trip_count']):
        unrolled.extend(plan['body'])
    splice_statements(tree, node, unrolled, clone_nodes=True)


# ---------------------------------------------------------------------------
# Pass 2: invariant if / bare-for guards
# ---------------------------------------------------------------------------

def fold_invariant_guards(session: Session, provider: HashProvider) -> int:
    tree = session.tree
    stats = Counter()

    # PHASE 1: Read-only Analysis (0 cache invalidations)
    actions = []

    for node in walk(tree.root, kinds={'IfStatement'}):
        test = node.get('test')
        if test is None:
            continue
        truth = evaluate_guard(test, provider, session, stats=stats)
        if truth is None:
            continue
        branch = node.get('consequent') if truth else node.get('alternate')
        statements = (
            list(branch.children_in('body'))
            if branch is not None and branch.kind == 'BlockStatement'
            else ([branch] if branch is not None else [])
        )
        actions.append((node, statements))

    for node in walk(tree.root, kinds={'ForStatement'}):
        # We only handle loops with NO update clause (update is None)
        if node.get('update') is not None:
            continue

        test = node.get('test')
        if test is None:
            continue

        truth = evaluate_guard(test, provider, session, stats=stats)
        if truth is None:
            continue

        init = node.get('init')
        init_stmts = []
        if init is not None:
            if init.kind == 'VariableDeclaration':
                init_stmts.append(clone(init))
            else:
                init_stmts.append(make('ExpressionStatement', expression=clone(init)))

        if not truth:
            # Condition is false: loop body never runs; keep only the init statement
            actions.append((node, init_stmts))
            continue

        # Condition is true: body executes
        body = node.get('body')
        stmts = (
            list(body.children_in('body'))
            if body is not None and body.kind == 'BlockStatement'
            else ([body] if body is not None else [])
        )
        if not stmts or stmts[-1] is None or \
                stmts[-1].kind not in ('ReturnStatement', 'ThrowStatement', 'BreakStatement'):
            continue

        # Splice: init statement followed by the terminating body statements
        actions.append((node, init_stmts + [clone(s) for s in stmts]))

    # PHASE 2: Mutation (Apply all splices at once)
    folded_count = 0
    for node, statements in actions:
        if not node.live:
            continue
        splice_statements(tree, node, statements, clone_nodes=True)
        folded_count += 1

    print(f' |->     guard stats: {dict(stats)}')
    return folded_count


# ---------------------------------------------------------------------------

def resolve_hash_guards(session: Session) -> None:
    print("--- Applying Transformer: resolve_hash_guards ---")
    tree = session.tree
    base_name = getattr(session, 'runtime_object', None)

    if not base_name:
        print(' |-> No runtime object.')
        return

    provider = find_hash_provider(tree, base_name)
    if provider is None:
        print(' |-> No murmur3-shaped hash provider.')
        return
    print(f' |-> Hash provider: {base_name}'
          f'[{provider.base_key!r}].{provider.method_name}')

    unrolled = unroll_counted_loops(session, provider)
    print(f' |->   Counter loops unrolled: {unrolled}')

    folded = fold_invariant_guards(session, provider)
    print(f' |->   Invariant guards folded: {folded}')
