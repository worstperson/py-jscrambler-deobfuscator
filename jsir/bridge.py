"""
Bridge to esprima2 / escodegen.

Parsing and printing JavaScript correctly is a solved problem that is no fun to
re-solve, so those stay at the edges. Everything between them works on the IR.

Two normalisations happen on the way in, and neither is reversible:

* `raw` is dropped. Literals carry a value; the printer regenerates the text.
  This is what makes \\x41-style escape obfuscation disappear for free.

* Numbers become floats and `null` becomes NULL. See jsir.values for why.

Comments are not carried. For obfuscated input that is a feature.
"""

import math

from . import values as V
from .schema import SCHEMA, ONE, MANY, slots_of, attributes_of
from .tree import Node, Tree, node as make


class BridgeError(Exception):
    pass


# esprima2 node kinds that deviate from ESTree. Normalised on the way in so the
# IR only ever sees the standard name.
_KIND_ALIASES = {
    'FieldDefinition': 'PropertyDefinition',
}


def _esprima_get(source, name, default=None):
    return getattr(source, name, default)


# ---------------------------------------------------------------------------
# esprima -> IR
# ---------------------------------------------------------------------------

def from_esprima(source):
    """Convert a parsed esprima program into a Tree."""
    root = _import_node(source)
    return Tree(root)


def _import_node(source):
    kind = _esprima_get(source, 'type')
    if kind is None:
        raise BridgeError(f'not an esprima node: {source!r}')
    kind = _KIND_ALIASES.get(kind, kind)
    if kind not in SCHEMA:
        raise BridgeError(f'no schema entry for {kind!r}')

    children = {}
    for slot, arity, _role in slots_of(kind):
        value = _esprima_get(source, slot)
        if value is None:
            children[slot] = [] if arity is MANY else None
            continue
        if arity is MANY:
            children[slot] = [_import_node(item) if item is not None else None
                              for item in value]
        else:
            children[slot] = _import_node(value)

    attrs = {}
    for name in attributes_of(kind):
        value = _esprima_get(source, name)
        if value is not None:
            attrs[name] = value

    if kind == 'Literal':
        attrs['value'] = _import_literal(source)
    elif kind == 'TemplateElement':
        holder = _esprima_get(source, 'value')
        attrs['cooked'] = _esprima_get(holder, 'cooked')
        attrs['raw'] = _esprima_get(holder, 'raw')
        attrs['tail'] = bool(_esprima_get(source, 'tail'))

    built = Node(kind, attrs, children, _esprima_get(source, 'range'))
    for slot, child, _role in built.iter_children():
        child.parent = built
        child.parent_slot = slot
    return built


def _import_literal(source):
    # Regex check must come first: RegexLiteral nodes have .value = None
    # (or a scanner RegExp object) and .regex set; the value field is useless.
    regex = _esprima_get(source, 'regex')
    if regex is not None:
        pattern = _esprima_get(regex, 'pattern') or ''
        flags = _esprima_get(regex, 'flags') or ''
        return V.JSRegExp(pattern, flags)

    value = _esprima_get(source, 'value')

    # Python None means one of two things here:
    #   - the literal is `null`      -> raw is 'null' or absent
    #   - the literal failed to parse (large integer etc.) -> raw holds the text
    # We distinguish by looking at raw. If raw is absent we trust it is null.
    if value is None:
        raw = _esprima_get(source, 'raw', 'null')
        if raw == 'null':
            return V.NULL
        # Numeric literals that lost precision: re-parse from raw.
        try:
            return float(raw)
        except (TypeError, ValueError):
            # Unknown non-null literal; keep as string so it round-trips.
            return str(raw) if raw is not None else V.NULL

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    # str, and anything else esprima produces
    return value


# ---------------------------------------------------------------------------
# IR -> esprima
# ---------------------------------------------------------------------------

class _Out:
    """A plain attribute bag for escodegen.

    escodegen accesses node fields as plain attributes, including ones that may
    not be set (e.g. `expr.regex` for every Literal, `stmt.alternate` for
    IfStatement). We return None for unknown attributes so escodegen's truthiness
    checks work correctly rather than throwing AttributeError.
    """

    def __getattr__(self, name):
        # Only called when normal attribute lookup fails.
        # Return None so `if expr.regex:` evaluates to False.
        if name.startswith('_'):
            raise AttributeError(name)
        return None

    def keys(self):
        return [key for key in self.__dict__ if key != 'type']

    def items(self):
        return [(key, value) for key, value in self.__dict__.items()
                if key != 'type']

    def __repr__(self):
        return f'<{getattr(self, "type", "?")}>'


def to_esprima(tree_or_node):
    root = tree_or_node.root if isinstance(tree_or_node, Tree) else tree_or_node
    return _export_node(root)


def _export_node(source):
    out = _Out()
    out.type = source.kind

    for slot, arity, _role in slots_of(source.kind):
        value = source.children.get(slot)
        if arity is MANY:
            setattr(out, slot, [_export_node(item) if item is not None else None
                                for item in (value or [])])
        else:
            setattr(out, slot, _export_node(value) if value is not None else None)

    for name in attributes_of(source.kind):
        if name in source.attrs:
            setattr(out, name, source.attrs[name])

    # Kind-specific fixups. These set attributes escodegen always reads
    # regardless of whether the IR node has them, so they must be explicit.
    kind = source.kind

    if kind == 'Literal':
        _export_literal(source, out)

    elif kind == 'TemplateElement':
        holder = _Out()
        holder.cooked = source.attr('cooked')
        holder.raw = source.attr('raw')
        out.value = holder
        out.tail = bool(source.attr('tail', False))

    elif kind == 'MemberExpression':
        out.computed = bool(source.attr('computed', False))
        out.optional = bool(source.attr('optional', False))

    elif kind in ('FunctionDeclaration', 'FunctionExpression'):
        out.generator = bool(source.attr('generator', False))
        out.expression = False
        out.isAsync = bool(source.attr('isAsync', False))

    elif kind == 'ArrowFunctionExpression':
        out.generator = False
        out.expression = (source.get('body') is not None and
                          source.get('body').kind != 'BlockStatement')
        out.isAsync = bool(source.attr('isAsync', False))

    elif kind == 'Program':
        out.sourceType = source.attr('sourceType', 'script')

    elif kind == 'Property':
        out.computed = bool(source.attr('computed', False))
        out.method = bool(source.attr('method', False))
        out.shorthand = bool(source.attr('shorthand', False))
        out.kind = source.attr('kind', 'init')

    elif kind in ('MethodDefinition', 'PropertyDefinition'):
        out.computed = bool(source.attr('computed', False))
        out.static = bool(source.attr('static', False))
        if kind == 'MethodDefinition':
            out.kind = source.attr('kind', 'method')

    elif kind in ('UnaryExpression', 'UpdateExpression'):
        out.prefix = bool(source.attr('prefix', True))

    elif kind == 'ForStatement':
        # escodegen reads .init, .test, .update whether or not they are set;
        # the slot export above handles that, but the `for (;;)` case
        # (all three None) still needs the attributes present.
        out.init = out.init  # already set by slot loop; this is a no-op
        out.test = out.test
        out.update = out.update

    return out


def _export_literal(source, out):
    """Fill the Literal-specific fields escodegen always inspects."""
    value = source.attr('value')

    if isinstance(value, V.JSRegExp):
        regex_holder = _Out()
        regex_holder.type = 'RegExp'          # for repr only; escodegen ignores type here
        regex_holder.pattern = value.pattern
        regex_holder.flags = value.flags
        out.regex = regex_holder
        # escodegen also reads .value and .raw on regex literals
        out.value = None
        out.raw = f'/{value.pattern}/{value.flags}'
        return

    # Non-regex: escodegen checks `if expr.regex:` first, which must be falsy.
    # _Out.__getattr__ returning None covers the attribute-absent case, but
    # being explicit here is safer if escodegen ever changes to `hasattr`.
    out.regex = None

    if value is V.NULL:
        out.value = None
        out.raw = 'null'
        return

    if value is V.UNDEF:
        raise BridgeError(
            'undefined is not a literal; represent it as Identifier("undefined") '
            'or UnaryExpression("void", Literal(0.0))')

    if isinstance(value, bool):
        out.value = value
        out.raw = 'true' if value else 'false'
        return

    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            raise BridgeError(
                f'{value} has no literal syntax; represent it as the expression '
                f'you mean: NaN -> Identifier("NaN"), Infinity -> '
                f'BinaryExpression("/", Literal(1), Literal(0)), etc.')
        # escodegen handles int-valued floats correctly when given a Python int.
        out.value = int(value) if value == int(value) and abs(value) < 1e21 else value
        out.raw = V.number_to_string(value)
        return

    if isinstance(value, str):
        out.value = value
        out.raw = None  # let escodegen choose quoting; this clears obfuscated raw
        return

    raise BridgeError(f'unexpected literal value type: {type(value).__name__} {value!r}')
