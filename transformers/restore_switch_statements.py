"""
restore_switch_statements — Reconstructs SwitchStatements from if-else chains.

Reconstructs `switch (x) { case ... }` blocks from chains of:
    if (x === A) { ... }
    else if (x === B) { ... }
    else { ... }

Also supports nested variants:
    if (x === A) { ... }
    else { if (x === B) { ... } }
"""

from jsir import Session, walk, node as make, clone
from transformers.helpers.ast_utils import is_equivalent

_TERMINATORS = frozenset((
    'ReturnStatement',
    'ThrowStatement',
    'BreakStatement',
    'ContinueStatement',
))


def restore_switch_statements(session: Session) -> None:
    print("--- Applying Transformer: restore_switch_statements ---")
    tree = session.tree
    count = 0

    for node in list(walk(tree.root, kinds={'IfStatement'})):
        if not node.live:
            continue

        # 1. Ensure this is the root of the chain (not an alternate sub-branch)
        parent = node.parent
        if parent is not None:
            if parent.kind == 'IfStatement' and parent.get('alternate') is node:
                continue
            if parent.kind == 'BlockStatement':
                grandparent = parent.parent
                if grandparent is not None and grandparent.kind == 'IfStatement' and grandparent.get('alternate') is parent:
                    continue

        # 2. Extract conditional chain (else if ... and else { if ... })
        current = node
        branches = []
        default_branch = None

        while current is not None:
            if current.kind == 'IfStatement':
                test = current.get('test')
                consequent = current.get('consequent')
                if test is None or consequent is None:
                    break

                branches.append({'test': test, 'consequent': consequent})
                alt = current.get('alternate')

                if alt is None:
                    current = None
                elif alt.kind == 'IfStatement':
                    current = alt
                elif alt.kind == 'BlockStatement':
                    body = alt.children_in('body')
                    if len(body) == 1 and body[0].kind == 'IfStatement':
                        current = body[0]
                    else:
                        default_branch = alt
                        current = None
                else:
                    default_branch = alt
                    current = None
            else:
                break

        # 3. Minimum 4 branches (greater than 3)
        if len(branches) <= 3:
            continue

        # 4. All tests must be strict equality (`===`)
        if not all(
            b['test'].kind == 'BinaryExpression' and b['test'].attr('operator') == '==='
            for b in branches
        ):
            continue

        # 5. Identify common discriminant candidate tested in all branches
        first_test = branches[0]['test']
        cand1 = first_test.get('left')
        cand2 = first_test.get('right')

        def get_cases_for_candidate(candidate):
            cases = []
            for b in branches:
                t = b['test']
                left = t.get('left')
                right = t.get('right')
                if is_equivalent(left, candidate):
                    cases.append({'case_expr': right, 'consequent': b['consequent']})
                elif is_equivalent(right, candidate):
                    cases.append({'case_expr': left, 'consequent': b['consequent']})
                else:
                    return None
            return cases

        matched_cases = get_cases_for_candidate(cand1)
        common_expr = cand1

        if matched_cases is None:
            matched_cases = get_cases_for_candidate(cand2)
            common_expr = cand2

        if matched_cases is None:
            continue

        # 6. Build SwitchCase nodes
        switch_cases = []
        for c in matched_cases:
            consequent = c['consequent']
            if consequent.kind == 'BlockStatement':
                statements = [clone(stmt) for stmt in consequent.children_in('body')]
            else:
                statements = [clone(consequent)]

            # Prevent fall-through if no explicit terminator exists
            terminates = bool(statements and statements[-1].kind in _TERMINATORS)
            if not terminates:
                statements.append(make('BreakStatement', label=None))

            switch_cases.append(make(
                'SwitchCase',
                test=clone(c['case_expr']),
                consequent=statements
            ))

        if default_branch is not None:
            if default_branch.kind == 'BlockStatement':
                default_stmts = [clone(stmt) for stmt in default_branch.children_in('body')]
            else:
                default_stmts = [clone(default_branch)]
            switch_cases.append(make('SwitchCase', test=None, consequent=default_stmts))

        # 7. Replace root IfStatement with the constructed SwitchStatement
        switch_stmt = make(
            'SwitchStatement',
            discriminant=clone(common_expr),
            cases=switch_cases
        )
        tree.replace(node, switch_stmt)
        count += 1

    print(f" |-> Restored {count} switch statement(s)")
