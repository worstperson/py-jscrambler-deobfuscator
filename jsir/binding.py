"""
Scopes and bindings.

Derived, with cooperative in-place tracking: built from the tree via a fast
whole-tree pass and updated in-place during common transformer mutations (folding
identifiers, removing dead expressions) without triggering whole-tree rescans.
"""

from .schema import (
    SCHEMA, ONE, MANY, VALUE, TARGET, LABEL, KEY,
    slots_of, FUNCTION_KINDS, VAR_SCOPE_KINDS, BLOCK_SCOPE_KINDS,
)


VAR_KINDS = frozenset(('var', 'function'))
LEXICAL_KINDS = frozenset(('let', 'const', 'class'))
PATTERN_KINDS = frozenset(('ArrayPattern', 'ObjectPattern', 'AssignmentPattern', 'RestElement'))

READ = 'read'
WRITE = 'write'
UPDATE = 'update'


class OrderedNodeSet:
    """An ordered collection of AST nodes supporting O(1) membership and removal."""
    __slots__ = ('_map',)

    def __init__(self, items=()):
        self._map = {item: None for item in items}

    def append(self, node):
        self._map[node] = None

    def remove(self, node):
        self._map.pop(node, None)

    def discard(self, node):
        self._map.pop(node, None)

    def __iter__(self):
        return iter(self._map)

    def __len__(self):
        return len(self._map)

    def __bool__(self):
        return bool(self._map)

    def __contains__(self, node):
        return node in self._map

    def __getitem__(self, index):
        return list(self._map)[index]

    def __repr__(self):
        return repr(list(self._map))


def _declares_loop_target(declarator):
    declaration = declarator.parent
    loop = declaration.parent if declaration is not None else None
    return (loop is not None and loop.kind in ('ForInStatement', 'ForOfStatement')
            and declaration.parent_slot == 'left')


def _pattern_usage(parent):
    """WRITE if the pattern receives a value here, None if it only declares."""
    if parent.kind == 'VariableDeclarator':
        return WRITE if parent.get('init') is not None or _declares_loop_target(parent) else None
    if parent.kind == 'AssignmentExpression' and parent.attr('operator', '=') == '=':
        return WRITE
    if parent.kind in ('ForInStatement', 'ForOfStatement'):
        return WRITE
    return None     # params, catch params


def _reference_pattern(pattern, usage, scope, scope_of, binding_of):
    kind = pattern.kind
    if kind == 'Identifier':
        if usage is WRITE:
            _record(pattern, WRITE, scope, binding_of)
        else:
            binding = scope.lookup(pattern.attr('name'))
            if binding is not None:
                binding_of[pattern.nid] = binding
    elif kind == 'AssignmentPattern':
        _reference_pattern(pattern.get('left'), usage, scope, scope_of, binding_of)
        right = pattern.get('right')
        if right is not None:
            _reference(right, scope, scope_of, binding_of)
    elif kind == 'RestElement':
        _reference_pattern(pattern.get('argument'), usage, scope, scope_of, binding_of)
    elif kind == 'ArrayPattern':
        for element in pattern.children_in('elements'):
            if element is not None:
                _reference_pattern(element, usage, scope, scope_of, binding_of)
    elif kind == 'ObjectPattern':
        for prop in pattern.children_in('properties'):
            if prop.kind == 'RestElement':
                _reference_pattern(prop.get('argument'), usage, scope, scope_of, binding_of)
                continue
            if prop.attr('computed', False):
                _reference(prop.get('key'), scope, scope_of, binding_of)
            _reference_pattern(prop.get('value'), usage, scope, scope_of, binding_of)
    else:
        # `[a.b] = x` and the like: a member target reads its object.
        _reference(pattern, scope, scope_of, binding_of)


class Binding:
    __slots__ = ('name', 'kind', 'scope', 'declaration', 'target',
                 'reads', 'writes', 'escapes', 'implicit_global')

    def __init__(self, name, kind, scope, declaration, target):
        self.name = name
        self.kind = kind
        self.scope = scope
        self.declaration = declaration    # the declarator / function / param owner
        self.target = target              # the Identifier that names it
        self.reads = OrderedNodeSet()
        self.writes = OrderedNodeSet()
        self.escapes = False              # read or written from a nested function
        self.implicit_global = False

    @property
    def is_read(self):
        return bool(self.reads)

    @property
    def is_written(self):
        return bool(self.writes)

    @property
    def constant(self):
        """Written at most once, counting the initialiser."""
        return len(self.writes) <= 1

    @property
    def promotable(self):
        """Safe to reason about as a local slot: no closure can observe an
        intermediate value, and nothing aliases it."""
        return (not self.escapes and not self.implicit_global and
                self.kind in ('var', 'let', 'const', 'param') and
                not self.scope.has_dynamic_scope)

    def __repr__(self):
        flags = ''.join(c for c, on in
                        (('E', self.escapes), ('P', self.promotable)) if on)
        return (f'Binding({self.name!r} {self.kind} r={len(self.reads)} '
                f'w={len(self.writes)} {flags})')


class Scope:
    __slots__ = ('node', 'kind', 'parent', 'children', 'bindings',
                 'has_dynamic_scope', 'function')

    def __init__(self, node, parent):
        self.node = node
        self.kind = node.kind
        self.parent = parent
        self.children = []
        self.bindings = {}
        self.has_dynamic_scope = bool(parent and parent.has_dynamic_scope)
        self.function = node if node.kind in FUNCTION_KINDS else (
            parent.function if parent else None)
        if parent is not None:
            parent.children.append(self)

    @property
    def is_var_scope(self):
        return self.kind in VAR_SCOPE_KINDS

    def var_scope(self):
        scope = self
        while not scope.is_var_scope and scope.parent is not None:
            scope = scope.parent
        return scope

    def lookup(self, name):
        scope = self
        while scope is not None:
            binding = scope.bindings.get(name)
            if binding is not None:
                return binding
            scope = scope.parent
        return None

    def declare(self, name, kind, declaration, target):
        holder = self.var_scope() if kind in VAR_KINDS else self
        existing = holder.bindings.get(name)
        if existing is not None:
            return existing
        binding = Binding(name, kind, holder, declaration, target)
        holder.bindings[name] = binding
        return binding

    def mark_dynamic(self):
        stack = [self]
        while stack:
            scope = stack.pop()
            scope.has_dynamic_scope = True
            stack.extend(scope.children)

    def __repr__(self):
        return f'<Scope {self.kind} {list(self.bindings)}>'


def _contains_structural_scope_nodes(node):
    """Whether a subtree contains nodes that introduce new scopes or variable declarations."""
    stack = [node]
    while stack:
        curr = stack.pop()
        if curr is None:
            continue
        k = curr.kind
        if k in FUNCTION_KINDS or k in BLOCK_SCOPE_KINDS or k == 'VariableDeclaration':
            return True
        for _slot, child, _role in curr.iter_children():
            stack.append(child)
    return False


class ScopeIndex:
    def __init__(self, root_scope, scope_of, binding_of, tree=None):
        self.root = root_scope
        self._scope_of = scope_of
        self._binding_of = binding_of
        self.tree = tree
        self.dirty = False

    def scope_of(self, node):
        current = node
        while current is not None:
            scope = self._scope_of.get(current.nid)
            if scope is not None:
                return scope
            current = current.parent
        return self.root

    def binding_of(self, identifier):
        """The binding an Identifier node resolves to, or None for a free name."""
        return self._binding_of.get(identifier.nid)

    def all_bindings(self):
        stack = [self.root]
        while stack:
            scope = stack.pop()
            yield from scope.bindings.values()
            stack.extend(scope.children)

    # -- cooperative in-place mutation handlers -----------------------------

    def mark_dirty(self):
        self.dirty = True

    def on_node_replaced(self, old_node, new_node):
        if self.dirty:
            return

        # If either side contains structural scope boundaries, flag for a fast whole-tree rebuild
        if _contains_structural_scope_nodes(old_node) or _contains_structural_scope_nodes(new_node):
            self.dirty = True
            return

        # Detach references from the replaced subtree
        self._detach_subtree(old_node)

        # Attach references from the new subtree
        parent = new_node.parent
        if parent is not None:
            scope = self.scope_of(parent)
            self._attach_subtree(new_node, scope)

    def on_node_removed(self, node):
        if self.dirty:
            return

        if _contains_structural_scope_nodes(node):
            if node.kind == 'VariableDeclaration':
                self._remove_var_declaration(node)
                return
            self.dirty = True
            return

        self._detach_subtree(node)

    def _detach_subtree(self, root):
        stack = [root]
        while stack:
            curr = stack.pop()
            if curr is None:
                continue
            if curr.kind == 'Identifier':
                b = self._binding_of.pop(curr.nid, None)
                if b is not None:
                    b.reads.discard(curr)
                    b.writes.discard(curr)
            for _slot, child, _role in curr.iter_children():
                stack.append(child)

    def _attach_subtree(self, root, scope):
        # Quick exit if replacement is literal or expression without identifiers
        stack = [(root, scope)]
        while stack:
            curr, s = stack.pop()
            if curr is None:
                continue

            s = self._scope_of.get(curr.nid, s)

            for slot, child, role in curr.iter_children():
                if role is TARGET and child.kind in PATTERN_KINDS:
                    _reference_pattern(child, _pattern_usage(curr), s, self._scope_of, self._binding_of)
                    continue
                if child.kind == 'Identifier':
                    usage = _usage_for(curr, slot, role, child)
                    if usage is not None:
                        _record(child, usage, s, self._binding_of)
                        continue
                stack.append((child, s))

    def _remove_var_declaration(self, decl_stmt):
        scope = self.scope_of(decl_stmt)
        for declarator in decl_stmt.children_in('declarations'):
            target = declarator.get('id')
            init = declarator.get('init')
            if init is not None:
                self._detach_subtree(init)
            if target is not None:
                for ident in _pattern_identifiers(target):
                    b = self._binding_of.pop(ident.nid, None)
                    if b is not None:
                        b.writes.discard(ident)
                        b.reads.discard(ident)
                        holder = b.scope
                        if holder.bindings.get(b.name) is b:
                            holder.bindings.pop(b.name, None)


def resolve(tree):
    scope_of = {}
    binding_of = {}
    root = Scope(tree.root, None)
    scope_of[tree.root.nid] = root

    _declare(tree.root, root, scope_of)
    _reference(tree.root, root, scope_of, binding_of)
    return ScopeIndex(root, scope_of, binding_of, tree)


# ---------------------------------------------------------------------------
# Pass 1: declarations
# ---------------------------------------------------------------------------

def _scope_for_children(node, scope, scope_of):
    if node.kind in BLOCK_SCOPE_KINDS and node is not scope.node:
        inner = Scope(node, scope)
        scope_of[node.nid] = inner
        return inner
    return scope


def _declare(node, scope, scope_of):
    kind = node.kind

    if kind in FUNCTION_KINDS:
        if kind == 'FunctionDeclaration':
            name_node = node.get('id')
            if name_node is not None:
                scope.declare(name_node.attr('name'), 'function', node, name_node)
        inner = Scope(node, scope)
        scope_of[node.nid] = inner
        if kind == 'FunctionExpression':
            name_node = node.get('id')
            if name_node is not None:
                inner.declare(name_node.attr('name'), 'function', node, name_node)
        if kind != 'ArrowFunctionExpression':
            inner.declare('arguments', 'param', node, None)
        for param in node.children_in('params'):
            for identifier in _pattern_identifiers(param):
                inner.declare(identifier.attr('name'), 'param', node, identifier)
            _declare_children(param, inner, scope_of)
        body = node.get('body')
        if body is not None:
            if body.kind == 'BlockStatement':
                scope_of[body.nid] = inner
                _declare_children(body, inner, scope_of)
            else:
                _declare(body, inner, scope_of)
        return

    if kind == 'VariableDeclaration':
        declaration_kind = node.attr('kind', 'var')
        for declarator in node.children_in('declarations'):
            target = declarator.get('id')
            for identifier in _pattern_identifiers(target):
                scope.declare(identifier.attr('name'), declaration_kind, declarator, identifier)
            _declare_children(declarator, scope, scope_of)
        return

    if kind in ('ClassDeclaration',):
        name_node = node.get('id')
        if name_node is not None:
            scope.declare(name_node.attr('name'), 'class', node, name_node)

    if kind == 'CatchClause':
        inner = Scope(node, scope)
        scope_of[node.nid] = inner
        param = node.get('param')
        if param is not None:
            for identifier in _pattern_identifiers(param):
                inner.declare(identifier.attr('name'), 'catch', node, identifier)
        body = node.get('body')
        if body is not None:
            scope_of[body.nid] = inner
            _declare_children(body, inner, scope_of)
        return

    if kind == 'WithStatement':
        scope.mark_dynamic()

    inner = _scope_for_children(node, scope, scope_of)
    _declare_children(node, inner, scope_of)


def _declare_children(node, scope, scope_of):
    for _slot, child, _role in node.iter_children():
        _declare(child, scope, scope_of)


def _pattern_identifiers(pattern):
    if pattern is None:
        return
    kind = pattern.kind
    if kind == 'Identifier':
        yield pattern
    elif kind == 'ObjectPattern':
        for prop in pattern.children_in('properties'):
            if prop.kind == 'RestElement':
                yield from _pattern_identifiers(prop.get('argument'))
            else:
                yield from _pattern_identifiers(prop.get('value'))
    elif kind == 'ArrayPattern':
        for element in pattern.children_in('elements'):
            yield from _pattern_identifiers(element)
    elif kind == 'AssignmentPattern':
        yield from _pattern_identifiers(pattern.get('left'))
    elif kind == 'RestElement':
        yield from _pattern_identifiers(pattern.get('argument'))


# ---------------------------------------------------------------------------
# Pass 2: references
# ---------------------------------------------------------------------------

def _reference(node, scope, scope_of, binding_of):
    scope = scope_of.get(node.nid, scope)

    if node.kind == 'CallExpression':
        callee = node.get('callee')
        if callee is not None and callee.kind == 'Identifier' and \
                callee.attr('name') == 'eval':
            scope.mark_dynamic()

    for slot, child, role in node.iter_children():
        if role is TARGET and child.kind in PATTERN_KINDS:
            _reference_pattern(child, _pattern_usage(node), scope, scope_of, binding_of)
            continue
        if child.kind == 'Identifier':
            usage = _usage_for(node, slot, role, child)
            if usage is not None:
                _record(child, usage, scope, binding_of)
                continue
        _reference(child, scope, scope_of, binding_of)


def _usage_for(parent, slot, role, child):
    """How this Identifier is used, or None if it is not a variable reference."""
    if role is LABEL:
        return None
    if role is KEY and not parent.attr('computed', False):
        return None
    if role is TARGET:
        kind = parent.kind
        if kind in ('VariableDeclarator',):
            if parent.get('init') is not None:
                return WRITE
            if _declares_loop_target(parent):
                return WRITE
            return None
        if kind == 'AssignmentExpression':
            return WRITE if parent.attr('operator', '=') == '=' else UPDATE
        if kind == 'UpdateExpression':
            return UPDATE
        if kind in ('ForInStatement', 'ForOfStatement'):
            return WRITE
        return None
    return READ


def _record(identifier, usage, scope, binding_of):
    name = identifier.attr('name')
    binding = scope.lookup(name)
    if binding is None:
        return
    binding_of[identifier.nid] = binding
    if usage in (READ, UPDATE):
        binding.reads.append(identifier)
    if usage in (WRITE, UPDATE):
        binding.writes.append(identifier)
    if scope.function is not binding.scope.function:
        binding.escapes = True
    if scope.has_dynamic_scope:
        binding.escapes = True


def declaration_writes(index):
    """Declarator initialisers count as writes but have no Identifier on the
    right-hand side to hang off, so they are attached here in one place."""
    for binding in index.all_bindings():
        declaration = binding.declaration
        if declaration is not None and declaration.kind == 'VariableDeclarator' \
                and declaration.get('init') is not None:
            target = binding.target
            if target is not None and target not in binding.writes:
                binding.writes.append(target)
