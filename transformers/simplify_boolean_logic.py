"""
simplify_boolean_logic -- purely structural boolean/branch simplification.
"""

from jsir import walk_post, node as make, clone, values as V
from transformers.helpers.ast_utils import hoisted_var_names, var_stub, splice_statements


def _literal_truthiness(n):
    if n is None:
        return None
    if n.kind == 'Literal':
        truth = V.truthiness(n.attr('value'))
        return None if truth is V.UNKNOWN else truth
    if n.kind == 'UnaryExpression' and n.attr('operator') == '!':
        inner = _literal_truthiness(n.get('argument'))
        return None if inner is None else not inner
    if n.kind == 'LogicalExpression':
        left = _literal_truthiness(n.get('left'))
        if left is None:
            return None
        operator = n.attr('operator')
        if (operator == '||' and left) or (operator == '&&' and not left):
            return left
        return _literal_truthiness(n.get('right'))
    if n.kind == 'ConditionalExpression':
        test = _literal_truthiness(n.get('test'))
        if test is None:
            return None
        return _literal_truthiness(n.get('consequent') if test else n.get('alternate'))
    return None


def _in_boolean_context(node):
    parent = node.parent
    if parent is None:
        return False
    slot = node.parent_slot
    if parent.kind in ('IfStatement', 'WhileStatement', 'DoWhileStatement',
                        'ForStatement', 'ConditionalExpression') and slot == 'test':
        return True
    if parent.kind == 'UnaryExpression' and parent.attr('operator') == '!' and slot == 'argument':
        return True
    if parent.kind == 'LogicalExpression' and slot in ('left', 'right'):
        return _in_boolean_context(parent)
    return False


def _promoted_statements(kept):
    if kept is None:
        return []
    if kept.kind != 'BlockStatement':
        return [clone(kept)]
    unsafe = {'ClassDeclaration', 'FunctionDeclaration'}
    for statement in kept.children_in('body'):
        if statement.kind in unsafe:
            return [clone(kept)]
        if statement.kind == 'VariableDeclaration' and statement.attr('kind', 'var') in ('let', 'const'):
            return [clone(kept)]
    return [clone(statement) for statement in kept.children_in('body')]


def _dead_branch_replacements(discarded, kept):
    replacements = []
    if discarded is not None:
        stub_names = hoisted_var_names(discarded)
        if stub_names:
            replacements.append(var_stub(stub_names))
    replacements.extend(_promoted_statements(kept))
    return replacements


def simplify_boolean_logic(session):
    print("--- Applying Transformer: simplify_boolean_logic ---")
    tree = session.tree

    for node in list(walk_post(tree.root)):
        if not node.live:
            continue
        kind = node.kind

        if kind == 'LogicalExpression':
            left, right = node.get('left'), node.get('right')
            operator = node.attr('operator')
            left_truth = _literal_truthiness(left)
            if left_truth is not None:
                if (operator == '||' and left_truth) or (operator == '&&' and not left_truth):
                    tree.replace(node, clone(left))
                else:
                    tree.replace(node, clone(right))
                continue

            right_truth = _literal_truthiness(right)
            identity = (operator == '&&' and right_truth is True) or \
                       (operator == '||' and right_truth is False)
            if identity and _in_boolean_context(node):
                tree.replace(node, clone(left))

        elif kind == 'ConditionalExpression':
            truth = _literal_truthiness(node.get('test'))
            if truth is None:
                continue
            branch = node.get('consequent') if truth else node.get('alternate')
            tree.replace(node, clone(branch))

        elif kind == 'IfStatement':
            truth = _literal_truthiness(node.get('test'))
            if truth is None:
                continue
            kept = node.get('consequent') if truth else node.get('alternate')
            discarded = node.get('alternate') if truth else node.get('consequent')
            splice_statements(tree, node, _dead_branch_replacements(discarded, kept))

        elif kind == 'WhileStatement':
            if _literal_truthiness(node.get('test')) is not False:
                continue
            splice_statements(tree, node, _dead_branch_replacements(node.get('body'), None))

        elif kind == 'DoWhileStatement':
            if _literal_truthiness(node.get('test')) is not False:
                continue
            splice_statements(tree, node, _promoted_statements(node.get('body')))

        elif kind == 'ForStatement':
            if _literal_truthiness(node.get('test')) is not False:
                continue
            replacements = []
            init = node.get('init')
            if init is not None:
                if init.kind == 'VariableDeclaration':
                    replacements.append(clone(init))
                else:
                    replacements.append(make('ExpressionStatement', expression=clone(init)))
            replacements.extend(_dead_branch_replacements(node.get('body'), None))
            splice_statements(tree, node, replacements)
