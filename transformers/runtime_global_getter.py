from jsir import Session, walk, node as make, clone
from transformers.helpers.ast_utils import is_equivalent
from jsir.schema import FUNCTION_KINDS


def get_node_value(node) -> str | None:
    """Extract string value from an Identifier or Literal node safely."""
    if not node:
        return None
    return node.attr('name') if node.kind == 'Identifier' else (node.attr('value') if node.kind == 'Literal' else None)


def is_global_getter(call_node) -> bool:
    callee = call_node.get('callee') if call_node.kind == 'CallExpression' else None
    if not callee or callee.kind != 'MemberExpression':
        return False

    if get_node_value(callee.get('object')) != 'Object' or get_node_value(callee.get('property')) != 'defineProperty':
        return False

    args = call_node.get('arguments') or []
    if len(args) < 3 or args[2].kind != 'ObjectExpression':
        return False

    for prop in (args[2].get('properties') or []):
        if get_node_value(prop.get('key')) == 'get':
            val = prop.get('value')
            stmts = (val.get('body').get('body') or []) if val and val.kind == 'FunctionExpression' else []
            return (
                len(stmts) == 1 and
                stmts[0].kind == 'ReturnStatement' and
                getattr(stmts[0].get('argument'), 'kind', None) == 'ThisExpression'
            )
    return False


def find_runtime_stub(runtime_root, proxy_name):
    """Searches strictly within the top-level statements of runtime_root for the stub function."""
    base_stmts = ()
    if runtime_root.kind in FUNCTION_KINDS:
        body = runtime_root.get('body')
        if body and body.kind == 'BlockStatement':
            base_stmts = body.children_in('body')
    elif runtime_root.kind in ('Program', 'BlockStatement'):
        base_stmts = runtime_root.children_in('body')

    for stmt in base_stmts:
        # Standard form: `function proxy_name() {}`
        if stmt.kind == 'FunctionDeclaration':
            fn_id = stmt.get('id')
            if fn_id and fn_id.kind == 'Identifier' and fn_id.attr('name') == proxy_name:
                return stmt

        # Alternate declaration form: `var proxy_name = function () {}`
        elif stmt.kind == 'VariableDeclaration':
            for decl in stmt.children_in('declarations'):
                d_id = decl.get('id')
                if d_id and d_id.kind == 'Identifier' and d_id.attr('name') == proxy_name:
                    return stmt

    return None


def handle_assignment(session: Session, assign) -> bool:
    left, right = assign.get('left'), assign.get('right')
    if not (left and left.kind == 'MemberExpression' and right and right.kind == 'CallExpression'):
        return False

    left_obj, callee = left.get('object'), right.get('callee')
    if not (left_obj and left_obj.kind == 'Identifier' and callee and callee.kind in {'FunctionExpression', 'ArrowFunctionExpression'}):
        return False

    proxy_name = left_obj.attr('name')
    if not any(is_global_getter(call) for call in walk(callee, kinds={'CallExpression'})):
        return False

    scope = session.scopes.scope_of(left_obj)
    binding = scope.lookup(proxy_name) if scope else None
    if not binding:
        return False

    print(" |-> found global getter function: " + str(proxy_name))

    global_name = '_global'
    session.runtime_object = global_name
    session.original_object = proxy_name
    session.global_getter = assign.parent
    session.oracle.runtime_object = global_name

    # Resolve the enclosing function/scope node where the runtime objects live
    runtime_root = assign.enclosing(*FUNCTION_KINDS)
    if not runtime_root and binding and binding.scope:
        scope_node = binding.scope.node
        if scope_node.kind in FUNCTION_KINDS or scope_node.kind == 'Program':
            runtime_root = scope_node

    session.runtime_root = runtime_root or session.tree.root

    root_desc = f"<{session.runtime_root.kind} #{session.runtime_root.nid}>"
    if session.runtime_root.kind == 'FunctionDeclaration':
        fn_id = session.runtime_root.get('id')
        if fn_id:
            root_desc += f" ({fn_id.attr('name')})"
    print(f" |-> runtime root: {root_desc}")

    print(" |-> replacing proxy references: " + str(len(binding.reads)))

    # Replace proxy references with runtime object identifier
    for ref in reversed(list(binding.reads)):
        parent = ref.parent
        is_alias = (
            parent and parent.kind == 'MemberExpression' and parent.get('object') is ref and
            is_equivalent(parent.get('property'), left.get('property')) and
            parent.attr('computed') == left.attr('computed')
        )
        session.tree.replace(parent if is_alias else ref, clone(make('Identifier', name=global_name)))

    # Locate the stub function at the base of runtime_root
    session.runtime_stub = find_runtime_stub(session.runtime_root, proxy_name)
    if session.runtime_stub:
        print(f" |-> found runtime stub: <{session.runtime_stub.kind} #{session.runtime_stub.nid}> ('{proxy_name}')")

        # Rename the stub declaration to global_name ('_global')
        if session.runtime_stub.kind == 'FunctionDeclaration':
            stub_id = session.runtime_stub.get('id')
            if stub_id and stub_id.live:
                session.tree.replace(stub_id, make('Identifier', name=global_name))
                print(f" |-> renamed runtime stub declaration to '{global_name}'")
        elif session.runtime_stub.kind == 'VariableDeclaration':
            for decl in session.runtime_stub.children_in('declarations'):
                d_id = decl.get('id')
                if d_id and d_id.live and d_id.kind == 'Identifier' and d_id.attr('name') == proxy_name:
                    session.tree.replace(d_id, make('Identifier', name=global_name))
                    print(f" |-> renamed runtime stub declaration to '{global_name}'")

    return True

# FIXME we don't need to rename, proper deob removes this getter
# maybe make a debug mode with resolved runtime function names?
def runtime_global_getter(session: Session) -> None:
    print("--- Applying Transformer: runtime_global_getter ---")
    # Default initializations
    session.runtime_root = session.tree.root
    session.runtime_stub = None

    for node in list(walk(session.tree.root, kinds={'AssignmentExpression'})):
        if handle_assignment(session, node):
            break
