"""
globals — Injected global object resolution, member property access, and alias tracking.
"""

from transformers.helpers.ast_utils import unwrap_expression
from transformers.helpers.literals import extract_literal_value, normalize_value_to_key

KNOWN_GLOBALS = frozenset({
    'globalThis', 'window', 'self', 'global',
    '_global', '__global', '$global',
})


def resolve_identifier(node, scopes=None):
    """Traces an identifier through single-write variable initializers."""
    curr = node
    seen = set()
    while curr and curr.kind == 'Identifier':
        name = curr.attr('name')
        if name in seen:
            break
        seen.add(name)

        scope = scopes.scope_of(curr) if scopes else None
        binding = scope.lookup(name) if scope else None
        if not binding:
            break

        raw_decl = getattr(binding, 'declaration', None) or getattr(binding, 'identifier', None) or getattr(binding, 'node', None)
        decl = raw_decl if raw_decl and raw_decl.kind == 'VariableDeclarator' else (
            raw_decl.parent if raw_decl and raw_decl.parent and raw_decl.parent.kind == 'VariableDeclarator' else None
        )
        if decl and decl.get('init'):
            curr = decl.get('init')
        else:
            break
    return curr


def is_runtime_object(node, session=None, include_known_globals: bool = True) -> bool:
    """True if node resolves to the session's runtime object or a standard global root."""
    if not node:
        return False
    node = unwrap_expression(node)
    if not node or node.kind != 'Identifier':
        return False

    name = node.attr('name')
    target_name = getattr(session, 'runtime_object', None)
    if target_name and name == target_name:
        return True
    if include_known_globals and name in KNOWN_GLOBALS:
        return True

    scopes = getattr(session, 'scopes', None)
    if scopes:
        resolved = resolve_identifier(node, scopes)
        if resolved and resolved is not node and resolved.kind == 'Identifier':
            resolved_name = resolved.attr('name')
            if target_name and resolved_name == target_name:
                return True
            if include_known_globals and resolved_name in KNOWN_GLOBALS:
                return True

    return False


def get_member_key(node, as_string: bool = True) -> str | int | float | None:
    """Extracts property key from a MemberExpression (both dot and computed notation)."""
    if not node or node.kind != 'MemberExpression':
        return None
    prop = node.get('property')
    if not prop:
        return None

    if not node.attr('computed', False):
        return prop.attr('name') if prop.kind == 'Identifier' else None

    prop = unwrap_expression(prop)
    if not prop:
        return None

    val, ok = extract_literal_value(prop)
    if ok:
        if as_string:
            return normalize_value_to_key(val)
        if isinstance(val, (int, float)) and not isinstance(val, bool) and val == int(val):
            return int(val)
        return val

    if prop.kind == 'Identifier':
        return prop.attr('name')

    return None


def is_global_member_access(node, session=None, include_known_globals: bool = True) -> bool:
    """Matches any property access directly on the global object (_global.foo or _global[123])."""
    if not node:
        return False
    node = unwrap_expression(node)
    if not node or node.kind != 'MemberExpression':
        return False
    obj = unwrap_expression(node.get('object'))
    return is_runtime_object(obj, session, include_known_globals)


def is_global_access(node, session=None, include_known_globals: bool = True) -> bool:
    """Matches either direct global reference or a global property access."""
    if not node:
        return False
    node = unwrap_expression(node)
    if not node:
        return False
    return (
        is_runtime_object(node, session, include_known_globals) or
        is_global_member_access(node, session, include_known_globals)
    )


def is_global_or_alias(node, session=None, global_aliases=None) -> bool:
    """True if node resolves to the global object or an established global self-alias."""
    if not node:
        return False
    node = unwrap_expression(node)
    if not node:
        return False

    if is_runtime_object(node, session):
        return True

    if node.kind == 'MemberExpression' and global_aliases:
        obj = unwrap_expression(node.get('object'))
        if is_global_or_alias(obj, session, global_aliases):
            key = get_member_key(node, as_string=True)
            if key and key in global_aliases:
                return True

    return False
