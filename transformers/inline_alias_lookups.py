from jsir import Session, walk, node as make
from jsir.schema import FUNCTION_KINDS


def is_alias_call_pattern(node) -> bool:
    if not node or node.kind != 'CallExpression':
        return False
    args = node.get('arguments') or []
    if len(args) not in (4, 5):
        return False

    a0, a1, a2, a3 = args[0], args[1], args[2], args[3]
    if not (a0 and a0.kind == 'Identifier'):
        return False
    if not (a1 and a1.kind == 'Literal' and isinstance(a1.attr('value'), str)):
        return False
    if not (a2 and a2.kind == 'Literal' and isinstance(a2.attr('value'), (int, float)) and not isinstance(a2.attr('value'), bool)):
        return False
    if not (a3 and a3.kind == 'Literal' and isinstance(a3.attr('value'), str)):
        return False

    return True


def is_referenced_identifier(node) -> bool:
    if not node or node.kind != 'Identifier':
        return False
    parent = node.parent
    if not parent:
        return True

    p_kind = parent.kind
    if p_kind in ('FunctionDeclaration', 'FunctionExpression', 'ArrowFunctionExpression'):
        if parent.get('id') is node:
            return False
        params = parent.get('params') or []
        if node in params:
            return False

    elif p_kind == 'VariableDeclarator':
        if parent.get('id') is node:
            return False

    elif p_kind in ('ClassDeclaration', 'ClassExpression'):
        if parent.get('id') is node:
            return False

    elif p_kind in ('Property', 'ObjectProperty'):
        if parent.get('key') is node and not parent.attr('computed', False) and not parent.attr('shorthand', False):
            return False

    elif p_kind in ('LabeledStatement', 'BreakStatement', 'ContinueStatement'):
        if parent.get('label') is node:
            return False

    elif p_kind == 'CatchClause':
        if parent.get('param') is node:
            return False

    elif p_kind in ('ImportSpecifier', 'ImportDefaultSpecifier', 'ImportNamespaceSpecifier'):
        if parent.get('local') is node or parent.get('imported') is node:
            return False

    elif p_kind == 'MemberExpression':
        if not parent.attr('computed', False) and parent.get('property') is node:
            return False

    return True


def should_rename_identifier(node, renames: dict) -> bool:
    if not node or node.kind != 'Identifier':
        return False
    name = node.attr('name')
    if name not in renames:
        return False

    parent = node.parent
    # Renaming non-computed member properties (e.g. obj.oldName -> obj.newName)
    if parent and parent.kind == 'MemberExpression' and not parent.attr('computed', False) and parent.get('property') is node:
        return True

    # Renaming general variable references
    return is_referenced_identifier(node)


def find_alias_lookups(session: Session) -> None:
    print("--- Applying Transformer: find_alias_lookups ---")
    tree = session.tree

    # -------------------------------------------------------------
    # PHASE 1: Analysis & Query (Read-Only)
    # -------------------------------------------------------------
    potential_wrappers = {}
    for call in walk(tree.root, kinds={'CallExpression'}):
        if is_alias_call_pattern(call):
            callee = call.get('callee')
            if callee and callee.kind == 'Identifier':
                name = callee.attr('name')
                if name not in potential_wrappers:
                    potential_wrappers[name] = []
                potential_wrappers[name].append(call)

    wrapper_name = None
    alias_lookups = []
    for name, calls in potential_wrappers.items():
        if len(calls) > len(alias_lookups):
            wrapper_name = name
            alias_lookups = calls

    if not wrapper_name or not alias_lookups:
        print(" |-> No aliased function found.")
        session.alias_table_function = None
        session.alias_lookups = []
        return

    # Crawl up from alias_lookups to find the enclosing runtime function node (e.g. function t3LoKV)
    alias_table_function = alias_lookups[0].enclosing(*FUNCTION_KINDS)

    session.alias_table_function = alias_table_function
    session.alias_lookups = alias_lookups

    if alias_table_function:
        fn_id = alias_table_function.get('id')
        fn_name = fn_id.attr('name') if fn_id else '<anonymous>'
        print(f" |-> Identified runtime alias table function: {fn_name}")


def inline_alias_lookups(session: Session) -> None:
    print("--- Applying Transformer: inline_alias_lookups ---")
    tree = session.tree

    alias_lookups = getattr(session, 'alias_lookups', None)
    if alias_lookups is None:
        find_aliased_functions(session)
        alias_lookups = getattr(session, 'alias_lookups', None)

    if not alias_lookups:
        print(" |-> No aliased calls to inline.")
        return

    renames = {}
    for call in alias_lookups:
        args = call.get('arguments') or []
        if len(args) >= 4:
            original_name = args[1].attr('value')
            new_name = args[3].attr('value')
            renames[new_name] = original_name

    print(f" |-> Renaming {len(renames)} nodes.")

    # Collect identifiers targeted for renaming
    renaming_targets = []
    for id_node in walk(tree.root, kinds={'Identifier'}):
        if should_rename_identifier(id_node, renames):
            new_name = renames[id_node.attr('name')]
            renaming_targets.append((id_node, new_name))

    # -------------------------------------------------------------
    # PHASE 2: Mutation (Write-Only)
    # -------------------------------------------------------------
    print(f" |-> Inlining {len(renaming_targets)} identifier occurrences...")
    for id_node, new_name in renaming_targets:
        if getattr(id_node, 'is_attached', True) and getattr(id_node, 'live', True):
            tree.replace(id_node, make('Identifier', name=new_name))
