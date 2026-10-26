"""
literals — Evaluation, inspection, and AST generation for JS constants and literals.
"""

import math
from jsir import node as make, values as V
from jsir.values import is_concrete
from transformers.helpers.ast_utils import unwrap_expression


def is_numeric_literal(node, allow_unary: bool = False) -> bool:
    """True if node is a number literal (int or float, not bool).
    If allow_unary is True, also matches `+N` or `-N`.
    """
    if not node:
        return False
    if node.kind == 'Literal':
        val = node.attr('value')
        return isinstance(val, (int, float)) and not isinstance(val, bool)
    if allow_unary and node.kind == 'UnaryExpression' and node.attr('operator') in ('+', '-'):
        return is_numeric_literal(node.get('argument'), allow_unary=False)
    return False


def is_int_literal(node) -> bool:
    """True if node is a literal integer."""
    if not node or node.kind != 'Literal':
        return False
    val = node.attr('value')
    return isinstance(val, (int, float)) and not isinstance(val, bool) and val == int(val)


def is_string_literal(node, unwrap: bool = True) -> bool:
    """True if node evaluates directly to a literal string."""
    if not node:
        return False
    if unwrap:
        node = unwrap_expression(node)
        if not node:
            return False
    return node.kind == 'Literal' and isinstance(node.attr('value'), str)


def is_boolean_literal(node, unwrap: bool = True) -> bool:
    """True if node evaluates directly to a literal boolean."""
    if not node:
        return False
    if unwrap:
        node = unwrap_expression(node)
        if not node:
            return False
    return node.kind == 'Literal' and isinstance(node.attr('value'), bool)


def extract_literal_value(node):
    """Extracts a concrete value from Literal, UnaryExpression (+, -, !, void),
    or standard global constant identifiers (undefined, NaN, Infinity).
    Returns (value, success: bool).
    """
    if not node:
        return None, False
    node = unwrap_expression(node)
    if not node:
        return None, False

    if node.kind == 'Literal':
        return node.attr('value'), True

    if node.kind == 'Identifier':
        name = node.attr('name')
        if name == 'undefined':
            return V.UNDEF, True
        if name == 'NaN':
            return float('nan'), True
        if name == 'Infinity':
            return float('inf'), True

    if node.kind == 'UnaryExpression':
        op = node.attr('operator')
        arg_val, ok = extract_literal_value(node.get('argument'))
        if ok:
            if op in ('+', '-'):
                if isinstance(arg_val, (int, float)) and not isinstance(arg_val, bool):
                    return (-arg_val if op == '-' else +arg_val), True
            elif op == '!':
                truth = V.truthiness(arg_val)
                if truth is not V.UNKNOWN:
                    return not truth, True
            elif op == 'void':
                return V.UNDEF, True

    return None, False


def normalize_value_to_key(val) -> str | None:
    """Coerces any JS value to its property key string equivalent."""
    if val is None or val is V.UNDEF:
        return 'undefined'
    if val is V.NULL:
        return 'null'
    if isinstance(val, bool):
        return 'true' if val else 'false'
    if isinstance(val, (int, float)):
        val_f = float(val)
        if math.isnan(val_f):
            return 'NaN'
        if math.isinf(val_f):
            return 'Infinity' if val_f > 0 else '-Infinity'
        if val_f.is_integer() and abs(val_f) < 1e21:
            return str(int(val_f))
        return V.number_to_string(val_f)
    if isinstance(val, str):
        return val
    return str(val)


def value_to_node(value):
    """Build an IR node for a concrete value, or None if it cannot be expressed
    as a literal. Callers must handle UNDEF and NaN identifiers as needed.
    Negative numbers are wrapped in UnaryExpression('-').
    Infinity is encoded as 1/0 to avoid variable shadowing.
    """
    if not is_concrete(value):
        return None
    if value is V.UNDEF:
        return make('Identifier', name='undefined')
    if value is V.NULL:
        return make('Literal', value=V.NULL)
    if isinstance(value, float):
        if math.isnan(value):
            return make('Identifier', name='NaN')
        if math.isinf(value):
            one = make('Literal', value=1.0)
            zero = make('Literal', value=0.0)
            expr = make('BinaryExpression', operator='/', left=one, right=zero)
            return make('UnaryExpression', operator='-', prefix=True, argument=expr) if value < 0 else expr
        if value < 0 or (value == 0.0 and math.copysign(1.0, value) < 0):
            magnitude = -value
            if magnitude == int(magnitude) and magnitude < 1e21:
                magnitude = int(magnitude)
            return make('UnaryExpression', operator='-', prefix=True,
                        argument=make('Literal', value=float(magnitude)))
        if value == int(value) and value < 1e21:
            return make('Literal', value=float(value))
        return make('Literal', value=value)
    if isinstance(value, V.JSRegExp):
        return make('Literal', value=value)
    return make('Literal', value=value)
