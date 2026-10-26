"""
remove_dead_code_after_terminator -- drop statements after a return/throw/
break/continue in the same statement list.
"""

from jsir import walk
from transformers.helpers.ast_utils import hoisted_var_names, var_stub

_TERMINATORS = ('ReturnStatement', 'ThrowStatement', 'BreakStatement', 'ContinueStatement')
_STATEMENT_LIST_SLOTS = (('Program', 'body'), ('BlockStatement', 'body'), ('SwitchCase', 'consequent'))


def remove_dead_code_after_terminator(session) -> int:
    print("--- Applying Transformer: remove_dead_code_after_terminator ---")
    tree = session.tree
    removed = 0

    for kind, slot in _STATEMENT_LIST_SLOTS:
        for owner in list(walk(tree.root, kinds={kind})):
            if not owner.live:
                continue
            statements = owner.children_in(slot)
            cut = next((i for i, stmt in enumerate(statements) if stmt.kind in _TERMINATORS), None)
            if cut is None or cut == len(statements) - 1:
                continue

            dead = statements[cut + 1:]
            removable = [stmt for stmt in dead if stmt.kind != 'FunctionDeclaration']

            stub_names = []
            for stmt in removable:
                stub_names.extend(n for n in hoisted_var_names(stmt) if n not in stub_names)

            anchor = statements[cut]
            for stmt in removable:
                if stmt.live:
                    tree.remove(stmt)
                    removed += 1
            if stub_names and anchor.live:
                tree.insert_after(anchor, [var_stub(stub_names)])

    print(f' |->   instances of dead code removed: {removed}')
    return removed
