from jsir import Session, walk, node as make, clone
from transformers.helpers.ast_utils import is_equivalent


def handle_assignment(session: Session, assign) -> bool:

    left = assign.get('left')

    proxy_name = session.original_object
    global_name = session.runtime_object


    # Collect all identifier nodes matching proxy_name directly from the AST walk
    refs = [node for node in walk(session.tree.root, kinds={'Identifier'}) if node.attr('name') == proxy_name]

    print(" |-> replacing proxy references: " + str(len(refs)))

    for ref in reversed(refs):
        parent = ref.parent
        is_alias = (
            parent and parent.kind == 'MemberExpression' and parent.get('object') is ref and
            is_equivalent(parent.get('property'), left.get('property')) and
            parent.attr('computed') == left.attr('computed')
        )
        session.tree.replace(parent if is_alias else ref, clone(make('Identifier', name=global_name)))

    return True

def prepare2(session: Session) -> None:
    print("--- Applying Transformer: prepare2 ---")

    for node in list(walk(session.tree.root, kinds={'AssignmentExpression'})):
        if handle_assignment(session, node):
            break
