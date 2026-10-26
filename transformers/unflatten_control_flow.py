"""
unflatten_control_flow — reconstruct structured control flow from a
switch-state-machine loop.

Finds `for (; state !== END; ) { switch (state) { ... } }` and
`while (state !== END) { switch (state) { ... } }` loops, builds a
CFG over the switch's cases, and rebuilds structured if/else and loop
statements from it via dominator analysis. Also detects the cyclical-array
generator pattern used to obfuscate array-index lookups, materialises the
array in-memory, and folds identity comparisons between resolved cells into
boolean literals.
"""

from jsir import Session, walk, node as make, clone, values as V
from jsir.flow import CFG, Block, dominators
from jsir.schema import FUNCTION_KINDS
from transformers.helpers.common import resolve_boolean_identity, member_key, is_numeric_literal, numeric_value


EXIT = 'exit'


class _ArrayStateIds:
    """Assigns each distinct cyclical-array cell referenced by one state
    machine a stable synthetic int, in order of first encounter.

    Some obfuscators express every state id as a lookup into a
    find_cyclical_array_generators array rather than a bare integer
    literal, so there is no real number to recover -- only the fact that
    two lookups land on the same cell (or don't) is meaningful. That's
    exactly what _resolve_boolean_identity already computes for the
    boolean-comparison fold, so this just adds identity -> synthetic-int
    numbering on top of it, memoised across every case label, the exit
    test, the initial value, and every state update for one loop, so they
    all agree on what "state 3" means even though the obfuscated program
    never wrote the number 3 anywhere.
    """

    def __init__(self, generated_arrays):
        self.generated_arrays = generated_arrays
        self._ids = {}
        self._next = 0

    def resolve(self, node):
        if not self.generated_arrays:
            return None
        matched, identity = resolve_boolean_identity(node, self.generated_arrays)
        if not matched:
            return None
        if identity not in self._ids:
            self._ids[identity] = self._next
            self._next += 1
        return self._ids[identity]


# ---------------------------------------------------------------------------
# State-machine body detection
# ---------------------------------------------------------------------------

def _find_flattened_body(session, generated_arrays, skip=frozenset()):
    """Find the loop state machine (ForStatement or WhileStatement) and
    return a descriptor, or None.

    Returns the first loop statement (in document order, excluding node ids
    in `skip`) whose shape matches _match_loop. debug=True prints
    every loop considered and why it was accepted/rejected.
    """
    debug = False #getattr(_find_flattened_body, 'debug', False)
    candidates = list(walk(session.tree.root, kinds={'ForStatement', 'WhileStatement'}))
    if debug:
        print(f" |-> unflatten_control_flow: scanning {len(candidates)} loop statement(s)")
    for node in candidates:
        if node.nid in skip:
            continue
        # Fresh per loop: two different loops -- or the same generated
        # array read by two different loops -- number their states
        # independently, matching how the original scoped
        # state_object_to_id_map to a single handle_for_statement_exit call.
        array_state = _ArrayStateIds(generated_arrays)
        desc = _match_loop(node, session, debug=debug, array_state=array_state)
        if desc is not None:
            if debug:
                print(f" |-> unflatten_control_flow: matched loop at nid={node.nid} "
                      f"with {len(desc['cases'])} case(s): {sorted(desc['cases'])}")
            return desc
    return None


def _match_loop(node, session, debug=False, array_state=None):
    """Match: for (; state !== END; ) { switch (state) { ... } }
    or: while (state !== END) { switch (state) { ... } }

    The canonical flattened loop keeps its exit condition in the loop
    test, not the switch body: `update` is empty (the state update happens
    inside the switch cases instead), and `test` is a `state !== END`
    comparison that stops the loop once a case sets state to END. A bare
    `for(;;)`, with exit handled entirely inside the switch (via
    `return`/`break`), also matches -- `_match_loop_exit_test` treats an
    absent test as EXIT. What's rejected is a *present* test that isn't
    this `state !== END` shape: we don't try to guess what an unrecognised
    exit condition means, so that case bails out of the match entirely
    rather than risk silently dropping a real exit path.
    """
    if node.kind == 'ForStatement':
        if node.get('update') is not None:
            if debug:
                print(f"   |-> for nid={node.nid}: rejected, has update")
            return None  # the state update lives in the switch body, not here
        test = node.get('test')
    elif node.kind == 'WhileStatement':
        test = node.get('test')
    else:
        return None

    body = node.get('body')
    switch = None
    if body is not None and body.kind == 'BlockStatement':
        stmts = body.children_in('body')
        if len(stmts) == 1 and stmts[0].kind == 'SwitchStatement':
            switch = stmts[0]
        elif debug:
            kinds = [s.kind for s in stmts]
            print(f"   |-> loop nid={node.nid}: rejected, body isn't [SwitchStatement] "
                  f"(got {kinds})")
    elif body is not None and body.kind == 'SwitchStatement':
        switch = body
    if switch is None:
        if debug and (body is None or body.kind not in ('BlockStatement', 'SwitchStatement')):
            print(f"   |-> loop nid={node.nid}: rejected, body isn't a "
                  f"BlockStatement or a bare SwitchStatement")
        return None
    disc = switch.get('discriminant')
    state_var_key = member_key(disc) if disc is not None else None
    if disc is None or disc.kind not in ('Identifier', 'MemberExpression') or state_var_key is None:
        if debug:
            print(f"   |-> loop nid={node.nid}: rejected, switch discriminant isn't "
                  f"a plain Identifier or a statically-keyed MemberExpression "
                  f"(kind={disc.kind if disc else None})")
        return None

    exit_value = _match_loop_exit_test(test, state_var_key, session, array_state)
    # exit_value is one of: an int state id (the sentinel the loop test
    # checks for), EXIT (test absent -- a bare for(;;), exit handled
    # entirely inside the switch body), or False (a test is present but
    # isn't the state-var !== END shape -- not our pattern, bail out
    # rather than guess at an exit condition we don't understand).
    if exit_value is False:
        if debug:
            print(f"   |-> loop nid={node.nid}: rejected, test present but "
                  f"not '<state var> !== <END>' shape")
        return None

    # Locate the initialiser: look for a var decl or assignment before the loop
    # that targets state_var_key.
    init_node = _find_initialiser(node, state_var_key, session)
    if init_node is None:
        if debug:
            print(f"   |-> loop nid={node.nid}: rejected, no initialiser found "
                  f"for the state variable among preceding siblings")
        return None

    all_case_labels = []
    cases = {}
    for case in switch.children_in('cases'):
        test_node = case.get('test')
        if test_node is None:
            all_case_labels.append('default')
            continue
        state_id = numeric_value(test_node, session, array_state)
        all_case_labels.append(f'{test_node.kind}->{state_id}')
        if state_id is None:
            continue
        # Unlabeled breaks belong to the switch, not the state machine's
        # own control flow; strip them before the state-update search
        # below, since `<update>; break;` is the standard shape and the
        # search only looks at the last statement.
        body_stmts = [stmt for stmt in case.children_in('consequent')
                      if not (stmt.kind == 'BreakStatement' and stmt.get('label') is None)]
        cases[state_id] = body_stmts

    if debug:
        print(f"   |-> loop nid={node.nid}: switch has {len(all_case_labels)} "
              f"case label(s): {all_case_labels}")
        print(f"   |-> loop nid={node.nid}: {len(cases)} recognised as numeric "
              f"state ids: {sorted(cases)}")

    if not cases:
        return None

    return {
        'for_node': node,             # Kept for compatibility with downstream code
        'loop_node': node,
        'switch': switch,
        'state_var_key': state_var_key,
        'init_node': init_node,
        'loop_exit_id': exit_value,   # state id that reaching-the-loop-test exits on, or EXIT
        'cases': cases,
        'array_state': array_state,   # same instance used to build the CFG and resolve the initial state
    }


def _match_loop_exit_test(test, state_var_key, session=None, array_state=None):
    """Recognise `<state var> !== <END>` as the loop's exit test.

    Returns:
      * an int  -- the state id that terminates the loop via the test
        (`END` resolves to a numeric constant, bare or masked)
      * EXIT    -- no test at all (`for(;;)`); exit is handled entirely
        inside the switch body, same as the original's bare-loop shape
      * False   -- a test is present but isn't this pattern; caller must
        reject the whole match, since we can't safely ignore an exit
        condition we don't understand
    """
    if test is None:
        return EXIT
    if test.kind != 'BinaryExpression' or test.attr('operator') != '!==':
        return False
    left, right = test.get('left'), test.get('right')
    if member_key(left) == state_var_key:
        end_node = right
    elif member_key(right) == state_var_key:
        end_node = left
    else:
        return False
    if end_node is not None and end_node.kind == 'Identifier' and end_node.attr('name') == 'undefined':
        return EXIT
    numeric = numeric_value(end_node, session, array_state) if end_node is not None else None
    if numeric is not None:
        return numeric
    return False


# Statement kinds that mark a control-flow boundary: if the state var's
# initialiser hasn't been found by the time the backward scan hits one of
# these, stop rather than keep walking into unrelated code above it.
_INIT_SCAN_BOUNDARY_KINDS = frozenset((
    'ForStatement', 'WhileStatement', 'DoWhileStatement', 'IfStatement',
    'ReturnStatement', 'SwitchStatement',
))


def _find_initialiser(loop_node, state_var_key, session):
    """Find where the state var's initial value is set.

    Checks a for-loop's own `init` clause first -- the common shape is
    `for (var a, e = 2; 9 !== e;)`, where the declaration is right there in
    the loop header, not a preceding statement. Only if that doesn't name
    the state var (or if it's a while loop) does this fall back to walking
    backwards through preceding sibling statements, and even then it stops
    at the first statement kind in _INIT_SCAN_BOUNDARY_KINDS rather than
    scanning arbitrarily far back.
    """
    if loop_node.kind == 'ForStatement':
        init = loop_node.get('init')
        if init is not None and _initialises(init, state_var_key):
            return init

    parent = loop_node.parent
    if parent is None:
        return None
    container = parent.children.get(loop_node.parent_slot)
    if not isinstance(container, list):
        return None
    try:
        idx = next(i for i, n in enumerate(container) if n is loop_node)
    except StopIteration:
        return None
    for i in range(idx - 1, -1, -1):
        candidate = container[i]
        if _initialises(candidate, state_var_key):
            return candidate
        if candidate.kind in _INIT_SCAN_BOUNDARY_KINDS:
            break
    return None


def _initialises(node, state_var_key):
    """True if `node` sets state_var_key to a real value: a VariableDeclaration
    with a matching declarator that has an `init`, or a `=` AssignmentExpression
    (bare, as in a for-loop's own init clause, or wrapped in an
    ExpressionStatement, as a preceding sibling would be)."""
    if node.kind == 'VariableDeclaration':
        for decl in node.children_in('declarations'):
            id_node = decl.get('id')
            if id_node is not None and member_key(id_node) == state_var_key \
                    and decl.get('init') is not None:
                return True
        return False
    expr = node.get('expression') if node.kind == 'ExpressionStatement' else node
    if expr is not None and expr.kind == 'AssignmentExpression' \
            and expr.attr('operator', '=') == '=':
        left = expr.get('left')
        if left is not None and member_key(left) == state_var_key:
            return True
    return False


# ---------------------------------------------------------------------------
# CFG over the switch cases
# ---------------------------------------------------------------------------

def _distribute_member_over_conditional(member):
    """`(test ? A : B)[i][j]...` -> a synthetic ConditionalExpression whose
    branches are `A[i][j]...` and `B[i][j]...`, freshly built from clones
    (never spliced into the real tree, and never adopts a live node's
    parent pointer by reference). Walks every stacked computed-integer
    index above the ConditionalExpression, not just one -- the original
    handled `(cond ? A : B)[i][j]`, not only a single trailing index. A
    masked state update sometimes hides behind an index chain shared by
    both arms of a ternary array pick -- `a = (7 === $ ?
    S99HFIc()[47][8] : S99HFIc()[1][31])[40]` -- rather than a bare `test
    ? A : B`. Distributing the index chain first lets it resolve exactly
    like the direct-ternary case below: each branch becomes a plain
    MemberExpression chain an _as_state_id/numeric_value call can walk.
    Returns None if nothing above a ConditionalExpression looks like a
    plain integer-indexed MemberExpression chain."""
    indices = []
    current = member
    while (current is not None and current.kind == 'MemberExpression' and
           current.attr('computed', False) and is_numeric_literal(current.get('property'))):
        indices.append(current.get('property'))
        current = current.get('object')
    if current is None or current.kind != 'ConditionalExpression':
        return None
    indices.reverse()  # collected outer-to-inner walking down; want source order

    test = current.get('test')
    consequent = current.get('consequent')
    alternate = current.get('alternate')
    if test is None or consequent is None or alternate is None:
        return None

    def _apply_indices(branch):
        acc = clone(branch)
        for idx in indices:
            acc = make('MemberExpression', object=acc, property=clone(idx),
                        computed=True, optional=False)
        return acc

    return make('ConditionalExpression', test=clone(test),
                 consequent=_apply_indices(consequent), alternate=_apply_indices(alternate))


def _build_case_cfg(desc, session=None, debug=False):
    """Build a `jsir.flow.CFG` over the switch's cases.

    Each case gets a `Block` using the state id itself as the block's bid
    -- not CFG.new_block's own counter -- so nothing here needs a
    state-id <-> bid translation layer; the rest of this module keeps
    addressing states the way it always has. `cfg.exit` is a real Block
    too, standing in for what used to be the bare EXIT sentinel as a dict
    key, so a case whose next state is EXIT is just an edge to it like any
    other, rather than a special-cased id.

    A block's `.kind` is 'return'|'jump'|'branch'|'invalid'; `.statements`
    holds the case body with the state-update stripped (never the test --
    see below); a 'jump' block has one successor labelled 'next', a
    'branch' block two labelled 'true'/'false', a 'return' block one
    labelled 'return' straight to cfg.exit. A 'branch' block's test
    expression lives in `.source`, not `.statements`: `source` already
    exists on Block purely for provenance (jsir.flow._Builder uses it the
    same way, e.g. an if-block's source is its `consequent`), so it's a
    clean place for "the expression this block's shape came from" without
    the test ever being mistaken for a real statement by _process_vars.

    desc['loop_exit_id'] is the state id (or EXIT) that terminates the
    loop via its own `!== END` test, as opposed to a `return`/`break`
    inside the switch body -- it is never a key in `cases` (a sentinel,
    not a real state), so every resolved next/true/false id is normalised
    through _resolve_target so such ids connect to cfg.exit consistently,
    the same way an explicit `undefined` update already does. Any other
    id that isn't a real case (a dangling target) also falls back to
    cfg.exit rather than being dropped.
    """
    cases = desc['cases']
    state_var_key = desc['state_var_key']
    loop_exit_id = desc.get('loop_exit_id', EXIT)
    array_state = desc.get('array_state')

    cfg = CFG(desc['switch'])
    cfg.exit = Block(EXIT, 'exit')
    cfg.blocks[EXIT] = cfg.exit
    for state_id in cases:
        cfg.blocks[state_id] = Block(state_id, 'invalid')

    def resolve_target(state_id):
        if state_id is not None and state_id == loop_exit_id:
            return EXIT
        return state_id

    def target_block(resolved_id):
        return cfg.blocks.get(resolved_id, cfg.exit)

    for state_id, body_stmts in cases.items():
        block = cfg.blocks[state_id]
        if not body_stmts:
            continue  # stays 'invalid', no successors

        last = body_stmts[-1]
        state_update_rhs = _get_state_update_rhs(body_stmts, state_var_key)

        if state_update_rhs is None:
            block.statements = body_stmts
            if last.kind == 'ReturnStatement':
                block.kind = 'return'
                cfg.connect(block, cfg.exit, 'return')
            continue  # else stays 'invalid'

        body_without_update = _strip_state_update(body_stmts, state_var_key)

        # Conditional: state_update_rhs is a ternary, or a MemberExpression
        # whose object is one, with the same index/key applied to both
        # branches (`(test ? A : B)[i]` -- distribute the index in first).
        distributed = None
        if state_update_rhs.kind == 'ConditionalExpression':
            distributed = state_update_rhs
        elif state_update_rhs.kind == 'MemberExpression':
            distributed = _distribute_member_over_conditional(state_update_rhs)

        if distributed is not None:
            test = distributed.get('test')
            true_val = resolve_target(_as_state_id(distributed.get('consequent'), session, array_state))
            false_val = resolve_target(_as_state_id(distributed.get('alternate'), session, array_state))
            if true_val is not None and false_val is not None and test is not None:
                block.kind = 'branch'
                block.statements = body_without_update
                block.source = test
                cfg.connect(block, target_block(true_val), 'true')
                cfg.connect(block, target_block(false_val), 'false')
                continue

        next_id = resolve_target(_as_state_id(state_update_rhs, session, array_state))
        if next_id is not None:
            block.kind = 'jump'
            block.statements = body_without_update
            cfg.connect(block, target_block(next_id), 'next')
        else:
            block.statements = body_stmts  # stays 'invalid'

    return cfg


def _get_state_update_rhs(stmts, state_var_key):
    """Find the state variable assignment in the last statement."""
    if not stmts:
        return None
    last = stmts[-1]
    if last.kind == 'ExpressionStatement':
        expr = last.get('expression')
        if expr and expr.kind == 'AssignmentExpression':
            left = expr.get('left')
            if left is not None and member_key(left) == state_var_key:
                return expr.get('right')
        if expr and expr.kind == 'SequenceExpression':
            exprs = expr.children_in('expressions')
            for e in reversed(exprs):
                if e.kind == 'AssignmentExpression':
                    left = e.get('left')
                    if left is not None and member_key(left) == state_var_key:
                        return e.get('right')
    if last.kind == 'VariableDeclaration':
        decls = last.children_in('declarations')
        if decls:
            last_decl = decls[-1]
            id_node = last_decl.get('id')
            if id_node is not None and member_key(id_node) == state_var_key:
                return last_decl.get('init')
    return None


def _strip_state_update(stmts, state_var_key):
    if not stmts:
        return []
    last = stmts[-1]
    without = list(stmts[:-1])
    if last.kind == 'ExpressionStatement':
        expr = last.get('expression')
        if expr and expr.kind == 'SequenceExpression':
            exprs = expr.children_in('expressions')
            # Keep everything except the state assignment.
            kept = [e for e in exprs
                    if not (e.kind == 'AssignmentExpression' and
                            e.get('left') is not None and
                            member_key(e.get('left')) == state_var_key)]
            if kept:
                if len(kept) == 1:
                    without.append(make('ExpressionStatement', expression=clone(kept[0])))
                else:
                    without.append(make('ExpressionStatement',
                                        expression=make('SequenceExpression',
                                                        expressions=[clone(e) for e in kept])))
    elif last.kind == 'VariableDeclaration':
        # `var x = 1, e = <next_state>;` -- the state update is only the
        # last declarator; the rest of the declaration still has to run.
        # A single declarator (just the update) is dropped whole, same as
        # `without = stmts[:-1]` above already does.
        decls = last.children_in('declarations')
        if len(decls) > 1:
            without.append(make('VariableDeclaration', kind=last.attr('kind', 'var'),
                                declarations=[clone(d) for d in decls[:-1]]))
    return without


def _as_state_id(node, session=None, array_state=None):
    """Resolve a state-update RHS (or a ternary branch of one) to a state
    id. `undefined` is the exit sentinel some obfuscators use directly as
    a state value (as opposed to the loop-test sentinel handled by
    _match_loop_exit_test / loop_exit_id -- two different idioms for the
    same "stop the machine" concept). Anything else numeric-but-not-a-
    bare-Literal (masked arithmetic, negation, a folded const reference,
    or a cyclical-array lookup) goes through numeric_value the same way
    case labels do -- see numeric_value's docstring for why that matters."""
    if node is None:
        return None
    if node.kind == 'Identifier' and node.attr('name') == 'undefined':
        return EXIT
    return numeric_value(node, session, array_state)


# ---------------------------------------------------------------------------
# Structuring
# ---------------------------------------------------------------------------

def _structure(initial_id, cfg, fwd_dom, post_dom, loop_info, hoisted_vars):
    """Reconstruct structured code from the CFG. Returns a list of IR nodes.

    fwd_dom/post_dom are jsir.flow.Dominators instances (forward and
    post-dominance respectively) over the case-id graph."""
    success = [True]
    memo = {}
    in_progress = set()

    def build(state_id, stop_id):
        if state_id == EXIT or state_id == stop_id:
            return []
        key = (state_id, stop_id)
        if key in memo:
            return memo[key]
        if state_id in in_progress:
            success[0] = False
            return None
        in_progress.add(state_id)
        result = _build_node(state_id, stop_id)
        in_progress.discard(state_id)
        if not success[0]:
            return None
        memo[key] = result
        return result

    def _build_node(state_id, stop_id):
        node = cfg.blocks.get(state_id)
        if not node or node.kind == 'invalid':
            success[0] = False
            return None

        body = _process_vars(node.statements, hoisted_vars)

        loop_data = loop_info.get(state_id)

        if loop_data:
            return _build_loop_node(state_id, node, body, loop_data, stop_id, build, success)

        if node.kind == 'return':
            return body

        if node.kind == 'jump':
            next_id = node.successor('next')
            if next_id == EXIT and stop_id is not None:
                return body + [make('BreakStatement', label=None)]
            rest = build(next_id, stop_id)
            if not success[0]:
                return None
            return body + rest

        # branch
        true_id = node.successor('true')
        false_id = node.successor('false')
        join_id = post_dom.idom.get(state_id)
        if join_id == EXIT or join_id == stop_id:
            join_id = None
        branch_stop = join_id if join_id is not None else stop_id

        true_body = build(true_id, branch_stop)
        false_body = build(false_id, branch_stop)
        if not success[0]:
            return None

        alternate = None
        if len(false_body) == 1 and false_body[0].kind == 'IfStatement':
            alternate = false_body[0]
        elif false_body:
            alternate = make('BlockStatement', body=false_body)

        if_stmt = make('IfStatement', test=clone(node.source),
                       consequent=make('BlockStatement', body=true_body),
                       alternate=alternate)
        tail = build(join_id, stop_id) if join_id is not None else []
        if not success[0]:
            return None
        return body + [if_stmt] + tail

    def _build_loop_node(state_id, node, body, loop_data, stop_id, build, success):
        loop_set = loop_data['nodes']
        latches = loop_data['latches']

        if node.kind == 'branch':
            true_id = node.successor('true')
            false_id = node.successor('false')
            true_in = true_id in loop_set
            false_in = false_id in loop_set
            if true_in and not false_in:
                entry_id, exit_id = true_id, false_id
                test_expr = clone(node.source)
            elif false_in and not true_in:
                entry_id, exit_id = false_id, true_id
                test_expr = make('UnaryExpression', operator='!', prefix=True,
                                 argument=clone(node.source))
            else:
                success[0] = False
                return None

            if entry_id == state_id:
                rest = build(exit_id, stop_id)
                if not success[0]:
                    return None
                return [make('DoWhileStatement', test=test_expr,
                             body=make('BlockStatement', body=body))] + rest
            else:
                loop_body = build(entry_id, state_id)
                rest = build(exit_id, stop_id)
                if not success[0]:
                    return None
                return body + [make('WhileStatement', test=test_expr,
                                    body=make('BlockStatement', body=loop_body))] + rest

        if node.kind == 'jump':
            if len(latches) != 1:
                success[0] = False
                return None
            latch_id = latches[0]

            if latch_id == state_id:
                return [make('WhileStatement',
                            test=make('Literal', value=True),
                            body=make('BlockStatement', body=body))]

            latch = cfg.blocks.get(latch_id)
            if not latch or latch.kind != 'branch':
                success[0] = False
                return None
            latch_true_id = latch.successor('true')
            latch_false_id = latch.successor('false')
            if latch_true_id == state_id:
                exit_id = latch_false_id
                test_expr = clone(latch.source)
            elif latch_false_id == state_id:
                exit_id = latch_true_id
                test_expr = make('UnaryExpression', operator='!', prefix=True,
                                 argument=clone(latch.source))
            else:
                success[0] = False
                return None

            middle = build(node.successor('next'), latch_id)
            if not success[0]:
                return None
            latch_body = _process_vars(latch.statements, hoisted_vars)
            do_body = body + middle + latch_body
            rest = build(exit_id, stop_id)
            if not success[0]:
                return None
            return [make('DoWhileStatement', test=test_expr,
                         body=make('BlockStatement', body=do_body))] + rest

        success[0] = False
        return None

    result = build(initial_id, None)
    return result if success[0] else None


def _process_vars(stmts, hoisted_vars):
    """Hoist var declarations to the top, collecting names for a shared decl."""
    out = []
    for stmt in stmts:
        if stmt.kind == 'VariableDeclaration' and stmt.attr('kind') == 'var':
            for decl in stmt.children_in('declarations'):
                id_node = decl.get('id')
                init = decl.get('init')
                if id_node is None or id_node.kind != 'Identifier':
                    # Destructuring target (ArrayPattern/ObjectPattern) --
                    # can't hoist a single name for it; keep the
                    # declaration as-is rather than drop the initialiser.
                    out.append(make('VariableDeclaration', kind='var', declarations=[clone(decl)]))
                    continue
                hoisted_vars.add(id_node.attr('name'))
                if init is not None:
                    out.append(make('ExpressionStatement',
                                    expression=make('AssignmentExpression',
                                                    operator='=',
                                                    left=make('Identifier', name=id_node.attr('name')),
                                                    right=clone(init))))
        else:
            out.append(clone(stmt))
    return out


# ---------------------------------------------------------------------------
# Main pass
# ---------------------------------------------------------------------------

def unflatten_control_flow(session: Session) -> None:
    """Unflatten every switch-state-machine loop in the tree, including
    nested ones, and fold cyclical-array boolean comparisons.

    The cyclical-array generator scan runs first, against the untouched
    tree: a generator assignment can live inside a case body, and
    unflattening would rewrite or hoist it before a later scan could match
    its shape.

    Loops are re-scanned from scratch after each one is structured, since
    structuring edits the tree (new statements, removed initialiser) and a
    batch of matches collected up front could resolve against stale
    surroundings.
    """
    #generated_arrays = session.generated_arrays
    print("--- Applying Transformer: unflatten_control_flow ---")

    generated_arrays = getattr(session, 'generated_arrays', {})

    tree = session.tree
    total = 0
    failed = 0
    skip = set()
    while True:
        desc = _find_flattened_body(session, generated_arrays, skip)
        if desc is None:
            break
        if _unflatten_one(session, desc):
            total += 1
        else:
            skip.add(desc['loop_node'].nid)
            failed += 1

    print(f' |-> unflatten_control_flow: unflattened {total} state machine(s)'
          + (f', {failed} unstructurable' if failed else ''))


def _format_state_var(state_var_key):
    if state_var_key[0] == 'id':
        return state_var_key[1]
    return f'{_format_state_var(state_var_key[0])}[{state_var_key[1]}]'


def _unflatten_one(session, desc) -> bool:
    """Structure a single matched state machine and splice it in. Returns
    True on success, False if structuring failed (caller should stop
    retrying rather than loop forever re-finding the same unstructurable
    loop)."""
    tree = session.tree
    #print(f" |-> unflatten_control_flow: {len(desc['cases'])} cases, "
    #      f"state_var='{_format_state_var(desc['state_var_key'])}'")

    cfg = _build_case_cfg(desc, session)
    node_ids = list(desc['cases'].keys())

    # Find the initial state.
    initial_id = _get_initial_state(desc, session)
    if initial_id is None or initial_id not in desc['cases']:
        print(' |-> unflatten_control_flow: cannot determine initial state; '
              'leaving this loop as-is')
        return False

    # Forward dominators (for if/else structuring), over the real CFG now --
    # cfg.successors/cfg.predecessors are used automatically.
    fwd = dominators(cfg, entry=initial_id)
    # Post-dominators (for join node location): same computation with the
    # graph reversed, by swapping which method plays which role.
    post = dominators(cfg, entry=cfg.exit.bid,
                       successors=cfg.predecessors, predecessors=cfg.successors)
    loop_info = fwd.natural_loops(node_ids, cfg.successors, cfg.predecessors)

    hoisted_vars = set()
    new_body = _structure(initial_id, cfg, fwd, post, loop_info, hoisted_vars)

    if new_body is None:
        print(' |-> unflatten_control_flow: structuring failed; leaving this loop as-is')
        return False

    nodes_to_insert = []
    if hoisted_vars:
        declarators = [make('VariableDeclarator',
                            id=make('Identifier', name=name), init=None)
                       for name in sorted(hoisted_vars)]
        nodes_to_insert.append(make('VariableDeclaration', kind='var',
                                    declarations=declarators))
    nodes_to_insert.extend(new_body)

    # Replace the loop with the unflattened code.
    loop_node = desc['loop_node']
    if not loop_node.live:
        return False
    tree.replace_many(loop_node, nodes_to_insert)

    # Remove the state-variable initialiser.
    init = desc['init_node']
    if init.live:
        tree.remove(init)

    #print(f' |-> unflatten_control_flow: produced {len(nodes_to_insert)} nodes')
    return True


def _get_initial_state(desc, session=None):
    init = desc['init_node']
    array_state = desc.get('array_state')
    if init.kind == 'VariableDeclaration':
        for decl in init.children_in('declarations'):
            id_node = decl.get('id')
            if id_node is not None and member_key(id_node) == desc['state_var_key']:
                init_val = decl.get('init')
                if init_val:
                    return numeric_value(init_val, session, array_state)
    expr = init.get('expression') if init.kind == 'ExpressionStatement' else init
    if expr is not None and expr.kind == 'AssignmentExpression':
        right = expr.get('right')
        if right:
            return numeric_value(right, session, array_state)
    return None
