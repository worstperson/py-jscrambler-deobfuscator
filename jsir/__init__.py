"""
jsir -- an IR for JavaScript deobfuscation.
"""

from . import values
from .binding import resolve, declaration_writes
from .bridge import from_esprima, to_esprima
from .ccp import Oracle, Solver, Facts, solve_all, pin_dead_write_bindings, \
    pin_settled_escaping_bindings
from .flow import build_cfg, dominators, function_bodies
from .schema import SCHEMA
from .tree import Tree, Node, node, clone, walk, walk_post

__all__ = [
    'Session', 'Oracle', 'Tree', 'Node', 'node', 'clone', 'walk', 'walk_post',
    'values', 'build_cfg', 'dominators',
]


class _Cache:
    def __init__(self, session, builder):
        self.session = session
        self.builder = builder
        self.revision = -1
        self.value = None

    def get(self):
        if self.revision != self.session.tree.revision:
            self.value = self.builder(self.session)
            self.revision = self.session.tree.revision
        return self.value


class Session:
    def __init__(self, tree, oracle=None):
        self.tree = tree
        self.oracle = oracle or Oracle()
        self.log = []
        self._scopes_cache = None
        self._scopes_revision = -1
        self._facts = _Cache(self, _build_facts)
        self.pinned_bindings = {}

        # Hook cooperative tree mutations
        self.tree.on_replace = self._on_tree_replace
        self.tree.on_remove = self._on_tree_remove
        self.tree.on_dirty = self._on_tree_dirty

    def _on_tree_replace(self, old_node, new_node):
        if self._scopes_cache is not None:
            self._scopes_cache.on_node_replaced(old_node, new_node)
            if not self._scopes_cache.dirty:
                self._scopes_revision = self.tree.revision + 1

    def _on_tree_remove(self, node):
        if self._scopes_cache is not None:
            self._scopes_cache.on_node_removed(node)
            if not self._scopes_cache.dirty:
                self._scopes_revision = self.tree.revision + 1

    def _on_tree_dirty(self):
        if self._scopes_cache is not None:
            self._scopes_cache.mark_dirty()

    # -- construction -------------------------------------------------------

    @classmethod
    def parse(cls, source, oracle=None):
        import esprima
        return cls(from_esprima(esprima.parseScript(source, {'tolerant': True})), oracle)

    @classmethod
    def from_ast(cls, ast, oracle=None):
        return cls(from_esprima(ast), oracle)

    def generate(self, **options):
        import escodegen
        return escodegen.generate(to_esprima(self.tree), **options)

    # -- analyses -----------------------------------------------------------

    @property
    def scopes(self):
        if self._scopes_cache is None or self._scopes_cache.dirty or \
                self._scopes_revision != self.tree.revision:
            self._scopes_cache = _build_scopes(self)
            self._scopes_cache.dirty = False
            self._scopes_revision = self.tree.revision
        return self._scopes_cache

    @property
    def facts(self):
        """{owner_nid: (Facts, CFG)} for every function body."""
        return self._facts.get()

    def facts_for(self, node):
        """The Facts covering a node, found by walking out to its function."""
        owner = node
        while owner is not None:
            entry = self.facts.get(owner.nid)
            if entry is not None:
                return entry[0]
            owner = owner.parent
        return None

    def constant_of(self, node):
        facts = self.facts_for(node)
        return facts.constant_of(node) if facts is not None else None

    def truth_of(self, node):
        facts = self.facts_for(node)
        return facts.truth_of(node) if facts is not None else values.UNKNOWN

    def binding_of(self, identifier):
        return self.scopes.binding_of(identifier)

    # -- running ------------------------------------------------------------

    def run(self, *passes):
        import time
        for step in passes:
            before_rev = self.tree.revision
            t0 = time.perf_counter()
            step(self)
            elapsed = time.perf_counter() - t0
            edits = self.tree.revision - before_rev
            self.log.append((getattr(step, '__name__', repr(step)), edits, elapsed))
        return self

    def run_until_stable(self, *passes, limit=8):
        for _ in range(limit):
            before = self.tree.revision
            self.run(*passes)
            if self.tree.revision == before:
                break
        return self

    def report(self):
        lines = [f"{'Pass Name':<35} | {'Edits':<8} | {'Time (s)':<8}"]
        lines.append("-" * 57)
        for name, edits, elapsed in self.log:
            lines.append(f"{name:<35} | {edits:<8} | {elapsed:<8.3f}")
        return "\n".join(lines)


def _build_facts(session, rounds=3):
    pinned = session.pinned_bindings
    results = solve_all(session.tree, session.scopes, session.oracle, pinned)
    for _ in range(rounds - 1):
        discovered = pin_dead_write_bindings(session.scopes, results)
        discovered.update(pin_settled_escaping_bindings(session.scopes, results))
        if not discovered or all(key in pinned for key in discovered):
            break
        pinned.update(discovered)
        results = solve_all(session.tree, session.scopes, session.oracle, pinned)
    return results


def _build_scopes(session):
    index = resolve(session.tree)
    declaration_writes(index)
    return index
