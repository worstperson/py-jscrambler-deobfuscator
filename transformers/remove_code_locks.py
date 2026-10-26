from collections import namedtuple
from jsir import Session, walk, walk_post
from transformers.helpers.ast_utils import (
    unwrap_expression,
    prune_sequence_expressions,
    is_iife_call,
    read_member_key_and_type,
)

ScoringFunction = namedtuple('ScoringFunction', 'base_name base_key has_name function_name')


class EnvironmentalScorer:
    __slots__ = ('base', 'key', 'method', 'has_name')

    def __init__(self, base, key, method, has_name=None):
        self.base = base
        self.key = key
        self.method = method
        if has_name is not None:
            self.has_name = has_name
        else:
            self.has_name = isinstance(key, str) and not key.isdigit()


def _normalize_scorer(raw) -> EnvironmentalScorer | None:
    if not raw:
        return None
    if isinstance(raw, dict):
        return EnvironmentalScorer(
            raw.get('base') or raw.get('base_name'),
            raw.get('key') or raw.get('base_key'),
            raw.get('method') or raw.get('function_name'),
            raw.get('has_name'),
        )
    if hasattr(raw, 'base_name') and hasattr(raw, 'base_key') and hasattr(raw, 'function_name'):
        return EnvironmentalScorer(raw.base_name, raw.base_key, raw.function_name, getattr(raw, 'has_name', None))
    if hasattr(raw, 'base') and hasattr(raw, 'key') and hasattr(raw, 'method'):
        return EnvironmentalScorer(raw.base, raw.key, raw.method, getattr(raw, 'has_name', None))
    if isinstance(raw, (tuple, list)):
        if len(raw) == 3:
            return EnvironmentalScorer(raw[0], raw[1], raw[2])
        if len(raw) >= 4:
            return EnvironmentalScorer(raw[0], raw[1], raw[3], raw[2])
    return None


def _match_key(prop_node, is_computed: bool, target_key):
    if prop_node is None:
        return False
    if isinstance(target_key, (int, float)):
        return (is_computed and prop_node.kind == 'Literal' and
                isinstance(prop_node.attr('value'), (int, float)) and
                float(prop_node.attr('value')) == float(target_key))
    if isinstance(target_key, str):
        if target_key.isdigit():
            if is_computed and prop_node.kind == 'Literal' and isinstance(prop_node.attr('value'), (int, float)):
                return float(prop_node.attr('value')) == float(target_key)
        if not is_computed and prop_node.kind == 'Identifier':
            return prop_node.attr('name') == target_key
        if is_computed and prop_node.kind == 'Literal':
            return prop_node.attr('value') == target_key
    return False


def _match_method(prop_node, is_computed: bool, target_method: str):
    if prop_node is None:
        return False
    if not is_computed and prop_node.kind == 'Identifier':
        return prop_node.attr('name') == target_method
    if is_computed and prop_node.kind == 'Literal':
        return prop_node.attr('value') == target_method
    return False


def _member_matches(node, scorer: EnvironmentalScorer) -> bool:
    if not node or node.kind != 'MemberExpression':
        return False
    if not _match_method(node.get('property'), node.attr('computed', False), scorer.method):
        return False
    obj = node.get('object')
    if not obj or obj.kind != 'MemberExpression':
        return False
    if not _match_key(obj.get('property'), obj.attr('computed', False), scorer.key):
        return False
    inner_obj = obj.get('object')
    return (inner_obj is not None and inner_obj.kind == 'Identifier' and
            inner_obj.attr('name') == scorer.base)


def _is_target(node, scorer: EnvironmentalScorer) -> bool:
    if not node:
        return False
    if node.kind == 'UnaryExpression' and node.attr('operator') == 'void':
        node = node.get('argument')
        if not node:
            return False
    if node.kind == 'CallExpression':
        callee = unwrap_expression(node.get('callee'))
        return callee is not None and _member_matches(callee, scorer)
    return _member_matches(node, scorer)


def environmental_scorer(session: Session) -> None:
    print("--- Applying Transformer: environmental_scorer ---")
    tree = session.tree
    base_name = getattr(session, 'runtime_object', None)
    if not base_name:
        print(' |-> No obfuscator runtime object found.')
        return

    scoring_fn = _find_scoring_function(tree, base_name)
    if scoring_fn is None:
        print(' |-> Failed to locate scoring function')
        return

    session.environmental_scorer = scoring_fn
    print(f' |-> Saved environmental scorer: {scoring_fn.base_name}[{scoring_fn.base_key}].'
          f'{scoring_fn.function_name}')


def _find_scoring_function(tree, base_name):
    for assign in walk(tree.root, kinds={'AssignmentExpression'}):
        left, right = assign.get('left'), assign.get('right')
        if left is None or right is None or left.kind != 'MemberExpression':
            continue
        obj = left.get('object')
        if obj is None or obj.kind != 'Identifier' or obj.attr('name') != base_name:
            continue
        key, has_name = read_member_key_and_type(left)
        if key is None:
            continue
        if not is_iife_call(right):
            continue
        function_name = _find_inner_scoring_assignment(right)
        if function_name is not None:
            return ScoringFunction(base_name, key, has_name, function_name)
    return None


def _find_inner_scoring_assignment(call_node):
    for inner in walk(call_node, kinds={'AssignmentExpression'}):
        if inner.attr('operator', '=') != '=':
            continue
        inner_right = inner.get('right')
        if inner_right is None or inner_right.kind not in ('FunctionExpression', 'ArrowFunctionExpression'):
            continue
        if not _returns_numeric_literal(inner_right):
            continue
        inner_left = inner.get('left')
        if inner_left is None or inner_left.kind != 'MemberExpression':
            continue
        prop = inner_left.get('property')
        if prop is None or prop.kind != 'Identifier':
            continue
        return prop.attr('name')
    return None


def _returns_numeric_literal(fn_node):
    for ret in walk(fn_node, kinds={'ReturnStatement'}):
        arg = ret.get('argument')
        if arg is not None and arg.kind == 'Literal' and isinstance(arg.attr('value'), float):
            return True
    return False


def remove_code_locks(session: Session) -> None:
    print("--- Applying Transformer: remove_code_locks ---")
    tree = session.tree

    raw_scorer = getattr(session, 'environmental_scorer', None)
    scorer = _normalize_scorer(raw_scorer)
    if scorer is None:
        print(" |-> No environmental scorer found on session. Skipping.")
        return

    print(f" |-> Found scoring function: {scorer.base}[{scorer.key}].{scorer.method}")

    sites_to_process = []
    for node in walk_post(tree.root):
        if node.kind == 'SequenceExpression':
            exprs = node.children_in('expressions')
            if any(_is_target(e, scorer) for e in exprs):
                sites_to_process.append(node)
        elif node.kind == 'ExpressionStatement':
            expr = node.get('expression')
            if expr is not None and _is_target(expr, scorer):
                sites_to_process.append(node)

    if not sites_to_process:
        print(" |-> No scoring function occurrences found.")
        return

    total_removed = 0
    for node in sites_to_process:
        if not node.live:
            continue

        if node.kind == 'SequenceExpression':
            total_removed += prune_sequence_expressions(tree, node, lambda e: not _is_target(e, scorer))
        elif node.kind == 'ExpressionStatement':
            expr = node.get('expression')
            if expr is not None and _is_target(expr, scorer):
                tree.remove(node)
                total_removed += 1

    print(f" |-> Removed {total_removed} scoring function references(s).")
