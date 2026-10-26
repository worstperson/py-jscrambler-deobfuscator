from jsir import Session, walk, node as make
from transformers.helpers.common import resolve_boolean_identity


def fold_boolean_comparisons(session: Session) -> None:
    """Fold `==`/`===`/`!=`/`!==` comparisons between two lookups into a
    known generated cyclical array into a plain boolean literal, e.g.:

        _global[X].arr()[2][5][17] == _global[X].arr()[2][0][5]

    Walks every BinaryExpression in the tree regardless of position (loop
    tests, if conditions, declarations, anywhere), repeated to a fixed
    point so a chain like `A == B != C` folds correctly: once `A == B`
    becomes a literal, the outer `!= C` no longer matches this pattern and
    is correctly left alone.
    """
    print("--- Applying Transformer: fold_boolean_comparisons ---")

    generated_arrays = session.generated_arrays
    if not generated_arrays:
        print(" |-> No cyclical arrays generated")
        return
    tree = session.tree

    count = 0
    for node in list(walk(tree.root, kinds=('BinaryExpression',))):
        if not node.live:
            continue
        operator = node.attr('operator')
        if operator not in ('==', '===', '!=', '!=='):
            continue

        left_matched, left_identity = resolve_boolean_identity(
            node.get('left'), generated_arrays)
        if not left_matched:
            continue
        right_matched, right_identity = resolve_boolean_identity(
            node.get('right'), generated_arrays)
        if not right_matched:
            continue

        are_equal = left_identity == right_identity
        result = are_equal if operator in ('==', '===') else not are_equal
        #print(f" |-> Folded obfuscated boolean comparison ({operator}) -> {result}")
        tree.replace(node, make('Literal', value=result))
        count += 1
    print(f" |-> Folded obfuscated boolean comparisons: {count}")

