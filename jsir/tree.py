"""
The tree.

Positions are identity-based. A node knows its parent and which slot of that
parent holds it; it does not store an index. Index is looked up by identity
when needed.

Ownership rules
---------------
* A node has at most one parent. `parent` is authoritative.
* Moving a node that is attached elsewhere requires `detach()` first, or
  `clone()` if the original stays. Edits raise EditError on aliasing.
* A node dies (`live = False`) only when it actually leaves the tree. Nodes
  that were re-adopted by another parent during an edit stay live.
* A node killed by `replace` / `replace_many` / `set_slot` records `forward`,
  which `Ref` follows to find its live successor.
"""

from itertools import count

from .schema import (
    SCHEMA, ONE, MANY, VALUE, TARGET, LABEL, KEY,
    slots_of, attributes_of, STATEMENT_KINDS,
)

_ids = count(1)


class Node:
    __slots__ = ('kind', 'nid', 'attrs', 'children', 'parent', 'parent_slot',
                 'live', 'origin', 'forward')

    def __init__(self, kind, attrs=None, children=None, origin=None):
        slots_of(kind)
        self.kind = kind
        self.nid = next(_ids)
        self.attrs = dict(attrs or {})
        self.children = children or {}
        self.parent = None
        self.parent_slot = None
        self.live = True
        self.origin = origin
        # Node | tuple[Node] | None. Meaningful only when dead:
        # None means "removed"; a Node/tuple is what replaced it.
        self.forward = None

    # -- reads -------------------------------------------------------------

    def get(self, slot):
        return self.children.get(slot)

    def attr(self, name, default=None):
        return self.attrs.get(name, default)

    def is_kind(self, *kinds):
        return self.kind in kinds

    @property
    def is_statement(self):
        return self.kind in STATEMENT_KINDS

    def child_slots(self):
        return slots_of(self.kind)

    def children_in(self, slot):
        value = self.children.get(slot)
        if value is None:
            return ()
        return tuple(value) if isinstance(value, list) else (value,)

    def iter_children(self):
        """(slot, child, role) for every present child, in schema order."""
        for slot, arity, role in slots_of(self.kind):
            value = self.children.get(slot)
            if value is None:
                continue
            if arity is MANY:
                for item in tuple(value):
                    if item is not None:
                        yield slot, item, role
            else:
                yield slot, value, role

    def index_in_parent(self):
        """Position within the parent's list slot, or None for a single slot."""
        if self.parent is None:
            return None
        container = self.parent.children.get(self.parent_slot)
        if not isinstance(container, list):
            return None
        for index, item in enumerate(container):
            if item is self:
                return index
        return None

    def ancestors(self):
        node = self.parent
        while node is not None:
            yield node
            node = node.parent

    def enclosing(self, *kinds):
        for node in self.ancestors():
            if node.kind in kinds:
                return node
        return None

    def enclosing_statement(self):
        node = self
        while node is not None:
            if node.is_statement:
                return node
            node = node.parent
        return None

    def __repr__(self):
        detail = ''
        if self.kind == 'Identifier':
            detail = f' {self.attrs.get("name")!r}'
        elif self.kind == 'Literal':
            detail = f' {self.attrs.get("value")!r}'
        elif 'operator' in self.attrs:
            detail = f' {self.attrs["operator"]!r}'
        return f'<{self.kind}{detail} #{self.nid}{"" if self.live else " DEAD"}>'


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def node(kind, /, **fields):
    schema_slots = {name: arity for name, arity, _role in slots_of(kind)}
    children = {}
    attrs = {}
    for key, value in fields.items():
        if key in schema_slots:
            if schema_slots[key] is MANY:
                value = list(value or [])
            children[key] = value
        else:
            attrs[key] = value
    built = Node(kind, attrs, children)
    for slot, child, _role in built.iter_children():
        _adopt(child, built, slot)
    return built


def _adopt(child, parent, slot):
    child.parent = parent
    child.parent_slot = slot


def clone(source, _seen=None):
    children = {}
    for slot, arity, _role in slots_of(source.kind):
        value = source.children.get(slot)
        if value is None:
            continue
        if arity is MANY:
            children[slot] = [clone(item) if item is not None else None for item in value]
        else:
            children[slot] = clone(value)
    copy = Node(source.kind, dict(source.attrs), children, source.origin)
    for slot, child, _role in copy.iter_children():
        _adopt(child, copy, slot)
    return copy


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------

def successors(start):
    """Live nodes that `start` became. [start] if still live, [] if removed."""
    out, stack = [], [start]
    while stack:
        n = stack.pop()
        if n.live:
            out.append(n)
        elif isinstance(n.forward, tuple):
            stack.extend(reversed(n.forward))
        elif n.forward is not None:
            stack.append(n.forward)
    return out


class Ref:
    """A handle on a node that survives replace / replace_many / set_slot."""
    __slots__ = ('_node',)

    def __init__(self, node):
        self._node = node

    def get(self):
        """The live node, or None if removed or split into several."""
        n = self._node
        if n.live:
            return n
        found = successors(n)
        if len(found) == 1:
            self._node = found[0]      # path compression
            return found[0]
        return None

    def all(self):
        """Every live successor (several after replace_many)."""
        return successors(self._node)

    @property
    def alive(self):
        return bool(successors(self._node))

    def __repr__(self):
        return f'<Ref -> {self.get()!r}>'


# ---------------------------------------------------------------------------
# The tree
# ---------------------------------------------------------------------------

REQUIRED_STATEMENT_SLOTS = {
    ('IfStatement', 'consequent'),
    ('WhileStatement', 'body'),
    ('DoWhileStatement', 'body'),
    ('ForStatement', 'body'),
    ('ForInStatement', 'body'),
    ('ForOfStatement', 'body'),
    ('LabeledStatement', 'body'),
    ('WithStatement', 'body'),
}


class EditError(Exception):
    pass


class Tree:
    """Owns a root node and the revision counter analyses key off."""

    def __init__(self, root):
        self.root = root
        self.revision = 0
        self._listeners = []
        self.on_replace = None
        self.on_remove = None
        self.on_dirty = None

    def on_change(self, callback):
        self._listeners.append(callback)

    def _touch(self):
        self.revision += 1
        for callback in self._listeners:
            callback(self.revision)

    # -- queries -----------------------------------------------------------

    def contains(self, target):
        node = target
        while node.parent is not None:
            node = node.parent
        return node is self.root and target.live

    def ref(self, target):
        return Ref(target)

    # -- mutation ----------------------------------------------------------

    def replace(self, target, replacement):
        """Swap one node for one node.

        `replacement` must be freshly built, detached, or a descendant of
        `target` (unwrapping). Clone it first if the original stays in the tree.
        """
        self._require_attached(target)
        if replacement is target:
            return replacement
        self._require_movable(replacement, within=(target,))
        parent, slot = target.parent, target.parent_slot
        container = parent.children.get(slot)
        if isinstance(container, list):
            container[self._locate(container, target)] = replacement
        else:
            parent.children[slot] = replacement
        _adopt(replacement, parent, slot)   # adopt before kill
        self._kill(target)
        target.forward = replacement

        # Notify cooperative scope listener if present
        if self.on_replace is not None:
            self.on_replace(target, replacement)
        elif self.on_dirty is not None:
            self.on_dirty()

        self._touch()
        return replacement

    def replace_many(self, target, replacements):
        """Swap one node for several. Only valid in a list slot."""
        self._require_attached(target)
        parent, slot = target.parent, target.parent_slot
        container = parent.children.get(slot)
        if not isinstance(container, list):
            raise EditError(
                f'replace_many on single slot {parent.kind}.{slot}; wrap the '
                f'replacements yourself and call replace')
        replacements = list(replacements)
        for item in replacements:
            if item is not target:
                self._require_movable(item, within=(target,))
        index = self._locate(container, target)
        container[index:index + 1] = replacements
        for item in replacements:
            _adopt(item, parent, slot)       # adopt before kill
        if any(item is target for item in replacements):
            # target stays in the tree; nothing to kill or forward.
            pass
        else:
            self._kill(target)
            target.forward = tuple(replacements)

        if self.on_dirty is not None:
            self.on_dirty()

        self._touch()

    def detach(self, target):
        """Unlink from the parent without killing. For moving a node elsewhere."""
        self._require_attached(target)
        self._unlink(target)

        if self.on_dirty is not None:
            self.on_dirty()

        self._touch()
        return target

    def remove(self, target):
        """Detach a node and kill it. `forward` stays None ("removed")."""
        self._require_attached(target)
        self._unlink(target)
        self._kill(target)

        if self.on_remove is not None:
            self.on_remove(target)
        elif self.on_dirty is not None:
            self.on_dirty()

        self._touch()

    def insert_before(self, anchor, additions):
        self._insert(anchor, additions, 0)

    def insert_after(self, anchor, additions):
        self._insert(anchor, additions, 1)

    def _insert(self, anchor, additions, offset):
        self._require_attached(anchor)
        parent, slot = anchor.parent, anchor.parent_slot
        container = parent.children.get(slot)
        if not isinstance(container, list):
            raise EditError(f'cannot insert beside single slot {parent.kind}.{slot}')
        additions = list(additions)
        for item in additions:
            if item is not None:
                self._require_movable(item)
        index = self._locate(container, anchor) + offset
        container[index:index] = additions
        for item in additions:
            _adopt(item, parent, slot)

        if self.on_dirty is not None:
            self.on_dirty()

        self._touch()

    def set_slot(self, parent, slot, value):
        arity = dict((name, arity) for name, arity, _r in slots_of(parent.kind)).get(slot)
        if arity is None:
            raise EditError(f'{parent.kind} has no slot {slot!r}')
        previous = parent.children.get(slot)
        stays = (parent, slot)
        if arity is MANY:
            value = list(value or [])
            old_items = [i for i in (previous or []) if i is not None]
            for item in value:
                if item is not None:
                    self._require_movable(item, within=old_items, stays=stays)
            # adopt before kill
            parent.children[slot] = value
            for item in value:
                if item is not None:
                    _adopt(item, parent, slot)
            for item in old_items:
                if not _in(value, item):
                    self._kill(item)
        else:
            if value is not None and value is not previous:
                self._require_movable(
                    value, within=(previous,) if previous is not None else ())
            parent.children[slot] = value
            if value is not None:
                _adopt(value, parent, slot)
            if previous is not None and previous is not value:
                self._kill(previous)
                previous.forward = value   # None if the slot was cleared

        if self.on_dirty is not None:
            self.on_dirty()

        self._touch()

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _locate(container, target):
        for index, item in enumerate(container):
            if item is target:
                return index
        raise EditError(f'{target!r} is not in its parent slot; tree is inconsistent')

    def _require_attached(self, target):
        if target.parent is None:
            raise EditError(f'{target!r} has no parent (root, or already detached)')
        if not target.live:
            raise EditError(f'{target!r} was already detached')

    @staticmethod
    def _require_movable(incoming, within=(), stays=None):
        """Reject edits that would leave `incoming` attached in two places.

        Allowed: detached nodes, descendants of any node in `within` (they are
        being unwrapped out of something that dies), and nodes that remain in
        the same (parent, slot) (`stays`).
        """
        if incoming.parent is None:
            return
        if stays is not None and incoming.parent is stays[0] \
                and incoming.parent_slot == stays[1]:
            return
        for ancestor in incoming.ancestors():
            for owner in within:
                if ancestor is owner:
                    return
        raise EditError(
            f'{incoming!r} is still attached under {incoming.parent!r}; '
            f'clone() it or detach() it first')

    def _unlink(self, target):
        parent, slot = target.parent, target.parent_slot
        container = parent.children.get(slot)
        if isinstance(container, list):
            del container[self._locate(container, target)]
        elif (parent.kind, slot) in REQUIRED_STATEMENT_SLOTS:
            placeholder = node('EmptyStatement')
            parent.children[slot] = placeholder
            _adopt(placeholder, parent, slot)
        else:
            parent.children[slot] = None
        target.parent = None
        target.parent_slot = None

    @staticmethod
    def _kill(target):
        """Mark `target` and its still-owned descendants dead.

        A descendant whose `parent` is no longer the node we reached it from
        has been re-adopted elsewhere and stays live, along with its subtree.
        Callers must adopt reused nodes *before* calling this.
        """
        target.live = False
        stack = [target]
        while stack:
            current = stack.pop()
            for _slot, child, _role in current.iter_children():
                if child.parent is not current:
                    continue
                child.live = False
                stack.append(child)
        target.parent = None
        target.parent_slot = None


def _in(items, target):
    return any(item is target for item in items)


# ---------------------------------------------------------------------------
# Traversal
# ---------------------------------------------------------------------------

def walk(root, kinds=None, enter_functions=True):
    """Pre-order."""
    stack = [root]
    while stack:
        current = stack.pop()
        if not current.live and current is not root:
            continue
        if kinds is None or current.kind in kinds:
            yield current
        if not current.live and current is not root:
            continue
        children = [child for _slot, child, _role in current.iter_children()]
        if not enter_functions:
            children = [c for c in children if c.kind not in SCHEMA or
                        c.kind not in ('FunctionDeclaration', 'FunctionExpression',
                                       'ArrowFunctionExpression')]
        stack.extend(reversed(children))


def walk_post(root):
    """Post-order."""
    for _slot, child, _role in tuple(root.iter_children()):
        if child.live:
            yield from walk_post(child)
    yield root
