import sys

from jsir import node as make, clone, walk, values as V
from lifter import vmloader as L
from lifter import vmdisasm as D
from lifter import vmexpr
from lifter.vmlift import lift_tree, program
from jsir import Session, Tree

from transformers.prepare2 import prepare2
from transformers.restore_local_variables import restore_local_variables
from transformers.restore_for_loops import restore_for_loops
from transformers.remove_code_locks import remove_code_locks
from transformers.inline_wrapper_functions import inline_wrapper_functions, resolve_wrapper_aliases
from transformers.inline_alias_lookups import inline_alias_lookups
from lifter.transformers.remove_timing_traps import remove_timing_traps

def lift_vm(session: Session) -> None:
    print("--- Applying Transformer: lift_vm ---")
    try:
        layout, analysis = L.build_layout(session)
    except Exception as e:
        print(f' |-> {e}')
        return
    payloads = L.find_payloads(session, analysis.interp)
    if not payloads:
        print(' |-> no call of the factory with a literal payload found')
        return

    #session.vm_function = analysis.interp.outer

    current = analysis.interp.outer
    guard = 0
    while guard < 10 and current is not None and current.kind != 'AssignmentExpression':  # I don't think this guard always works, inf loop, FIXME
        current = current.parent
        guard += 1
    if current is not None:
        left = current.get('left')
        if left is not None and left.kind == 'Identifier':
            session.dce_keep.append(current.get('left').attr('name'))
            session.vm_function = current.parent

    call_node, payload = payloads[0]
    ctx, root, results = lift_tree(session, layout, analysis, call_node, payload, 'production')
    prog = program(ctx, results, True)

    vmsession = Session(Tree(prog))

    vmsession.runtime_object = session.runtime_object
    vmsession.original_object = session.original_object
    vmsession.wrapper_aliases = session.wrapper_aliases
    vmsession.environmental_scorer = session.environmental_scorer
    vmsession.alias_lookups = session.alias_lookups
    vmsession.oracle.runtime_object = None

    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    vmexpr.simplify(vmsession)
    remove_timing_traps(vmsession)
    prepare2(vmsession)
    restore_local_variables(vmsession)
    resolve_wrapper_aliases(vmsession)
    remove_code_locks(vmsession)
    inline_alias_lookups(vmsession)

    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    #print(vmsession.generate())

    # Extract the statement(s) inside the lifted Program root (not the Program node itself)
    body_stmts = [clone(s) for s in vmsession.tree.root.children_in('body')]

    tree = session.tree
    parent = call_node.parent

    # Case A: var W5uQlFZ = l0(...)
    if parent.kind == 'VariableDeclarator':
        decl_stmt = parent.parent
        # Replace the `var ... = l0(...)` statement with the lifted function declarations
        tree.replace_many(decl_stmt, body_stmts)

    # Case B: U_h(...)()
    elif parent.kind == 'CallExpression' and parent.get('callee') is call_node:
        expr_stmt = parent.parent
        # Replace the outer call statement with the self-executing IIFE statement
        tree.replace_many(expr_stmt, body_stmts)

    # Fallback: replace call expression directly
    else:
        if len(body_stmts) == 1 and body_stmts[0].kind == 'ExpressionStatement':
            tree.replace(call_node, body_stmts[0].get('expression'))
        else:
            stmt = call_node
            while stmt and not stmt.is_statement:
                stmt = stmt.parent
            if stmt:
                tree.replace_many(stmt, body_stmts)

    for node, lifted, ids in results:
        for n in lifted.notes:
            print(f'// NOTE {node.factory}: {n}', file=sys.stderr)
    print(f' |-> {len(results)} functions lifted, {sum(len(l.blocks) for _, l, _ in results)} blocks', file=sys.stderr)
