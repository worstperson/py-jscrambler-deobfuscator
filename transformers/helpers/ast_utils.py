"""
ast_utils — AST statement splicing, hoisting, and tree-manipulation helpers.
"""

from jsir import walk, clone, node as make
from jsir.binding import _pattern_identifiers


def hoisted_var_names(subtree) -> list[str]:
    """Collect all 'var' names declared in `subtree`, without descending
    into nested functions.
    """
    names = []
    seen = set()
    for declaration in walk(subtree, kinds={'VariableDeclaration'}, enter_functions=False):
        if declaration.attr('kind', 'var') != 'var':
            continue
        for declarator in declaration.children_in('declarations'):
            for identifier in _pattern_identifiers(declarator.get('id')):
                name = identifier.attr('name')
                if name not in seen:
                    seen.add(name)
                    names.append(name)
    return names


def var_stub(names: list[str]):
    """Build a bare `var a, b, c;` statement with no initializers to preserve
    hoisted bindings when discarding dead branches.
    """
    if not names:
        return None
    return make(
        'VariableDeclaration',
        kind='var',
        declarations=[
            make('VariableDeclarator', id=make('Identifier', name=name), init=None)
            for name in names
        ],
    )


def splice_statements(tree, target, replacements: list, clone_nodes: bool = False) -> None:
    """Safely install `replacements` in place of `target` statement.

    Handles removal, 1-to-1 replacement, `replace_many` in list containers,
    or synthesizing a `BlockStatement` when replacing inside a single-statement slot
    (e.g., a braceless `if` consequent).
    """
    if clone_nodes:
        replacements = [clone(s) for s in replacements]

    if not replacements:
        tree.remove(target)
    elif len(replacements) == 1:
        tree.replace(target, replacements[0])
    elif isinstance(target.parent.children.get(target.parent_slot), list):
        tree.replace_many(target, replacements)
    else:
        tree.replace(target, make('BlockStatement', body=replacements))


def unwrap_expression(node):
    """Recursively unwrap ParenthesizedExpression, ChainExpression, and trailing
    SequenceExpression expressions down to the core expression.
    """
    curr = node
    while curr:
        if curr.kind in ('ParenthesizedExpression', 'ChainExpression'):
            curr = curr.get('expression')
        elif curr.kind == 'SequenceExpression':
            exprs = curr.children_in('expressions') if hasattr(curr, 'children_in') else (curr.get('expressions') or [])
            curr = exprs[-1] if exprs else None
        else:
            break
    return curr


def is_equivalent(n1, n2, scopes=None, visited=None) -> bool:
    """Check structural equivalence between two AST expressions.
    Recursively compares Identifiers (with optional scope tracing), Literals,
    MemberExpressions, UnaryExpressions, and BinaryExpressions.
    """
    if n1 is n2:
        return True
    if not (n1 and n2) or n1.kind != n2.kind:
        return False

    if visited is None:
        visited = set()
    pair = (id(n1), id(n2))
    if pair in visited:
        return True
    visited.add(pair)

    if n1.kind == 'Identifier':
        if n1.attr('name') == n2.attr('name'):
            return True
        if not scopes:
            return False
        from transformers.helpers.globals import resolve_identifier
        r1 = resolve_identifier(n1, scopes)
        r2 = resolve_identifier(n2, scopes)
        if r1 is n1 and r2 is n2:
            return False
        return is_equivalent(r1, r2, scopes, visited)

    if n1.kind == 'Literal':
        v1, v2 = n1.attr('value'), n2.attr('value')
        if v1 == v2:
            return True
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            return not isinstance(v1, bool) and not isinstance(v2, bool) and float(v1) == float(v2)
        return str(v1) == str(v2)

    if n1.kind == 'MemberExpression':
        if n1.attr('computed', False) != n2.attr('computed', False):
            return False
        return (
            is_equivalent(n1.get('object'), n2.get('object'), scopes, visited) and
            is_equivalent(n1.get('property'), n2.get('property'), scopes, visited)
        )

    if n1.kind == 'UnaryExpression':
        return (
            n1.attr('operator') == n2.attr('operator') and
            is_equivalent(n1.get('argument'), n2.get('argument'), scopes, visited)
        )

    if n1.kind == 'BinaryExpression':
        return (
            n1.attr('operator') == n2.attr('operator') and
            is_equivalent(n1.get('left'), n2.get('left'), scopes, visited) and
            is_equivalent(n1.get('right'), n2.get('right'), scopes, visited)
        )

    return False


def references_name(node, name: str) -> bool:
    """True if `node` or any of its descendants is an Identifier matching `name`."""
    if not node:
        return False
    for n in walk(node, kinds={'Identifier'}):
        if n.attr('name') == name:
            return True
    return False


def flatten_sequence(expr) -> list:
    """Recursively flatten nested SequenceExpressions: `(a, (b, c))` -> `[a, b, c]`."""
    if expr is None or expr.kind != 'SequenceExpression':
        return [expr] if expr is not None else []
    flattened = []
    exprs = expr.children_in('expressions') if hasattr(expr, 'children_in') else (expr.get('expressions') or [])
    for item in exprs:
        flattened.extend(flatten_sequence(item))
    return flattened


def prune_sequence_expressions(tree, seq_node, keep_predicate) -> int:
    """Filters expressions in `seq_node` according to `keep_predicate`.
    Collapses or removes the sequence if `len(kept) <= 1`. Returns number of removed items.
    """
    if not seq_node or not seq_node.live or seq_node.kind != 'SequenceExpression':
        return 0

    exprs = seq_node.children_in('expressions') if hasattr(seq_node, 'children_in') else (seq_node.get('expressions') or [])
    kept = [e for e in exprs if keep_predicate(e)]
    removed_count = len(exprs) - len(kept)
    if removed_count == 0:
        return 0

    if len(kept) == 1:
        tree.replace(seq_node, clone(kept[0]))
    elif len(kept) == 0:
        parent = seq_node.parent
        if parent is not None and parent.kind == 'ExpressionStatement':
            tree.remove(parent)
        else:
            tree.replace(seq_node, make('Identifier', name='undefined'))
    else:
        tree.set_slot(seq_node, 'expressions', kept)

    return removed_count


def is_iife_call(node) -> bool:
    """Matches `(function(...) { ... })()` or `((...) => { ... })()`."""
    if not node or node.kind != 'CallExpression':
        return False
    callee = unwrap_expression(node.get('callee'))
    return callee is not None and callee.kind in ('FunctionExpression', 'ArrowFunctionExpression')


def read_member_key_and_type(member_node):
    """Reads a MemberExpression property, returning `(key, is_identifier_name)`.
    Key is float for numeric literals, str for identifiers (non-computed) or strings.
    """
    if not member_node or member_node.kind != 'MemberExpression':
        return None, False
    prop = member_node.get('property')
    if prop is None:
        return None, False
    if prop.kind == 'Literal':
        val = prop.attr('value')
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val), False
        if isinstance(val, str):
            return val, False
    if prop.kind == 'Identifier' and not member_node.attr('computed', False):
        return prop.attr('name'), True
    return None, False
