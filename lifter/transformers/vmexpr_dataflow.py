"""
vmexpr_dataflow -- data-flow passes: web splitting, copy propagation, CCP folding, dead stores, declarations.

Part of vmexpr (see vmexpr.py for the pipeline and the soundness notes).
"""

import re
import sys

from jsir import Session, Tree, node as make, clone, walk, values as V
from jsir.schema import FUNCTION_KINDS, TARGET, KEY, LABEL
from lifter.vmstructure import _negate_expr
from lifter.transformers.vmexpr_liveness import (
    _Liveness,
    _escapes,
)
from lifter.transformers.vmexpr_util import (
    assignment_of,
    body_lists,
    declare,
    fresh,
    ident,
    leaf_functions,
    literal_node,
    names_written,
    pure,
    scan,
    tracked_bindings,
    used_names,
)


def split_webs(session):
    """A def->use pair of a reused slot becomes a variable of its own:

        s0 = a; ... s0[s1] = s2;      s0 reused later for something else
        =>  w0 = a; ... w0[s1] = s2;

    valid when the def reaches reads in exactly one later statement of the same
    list and the variable is dead after it (and nothing jumps out in between).
    The fresh variable is single-write/single-read, which is what inline_temps
    needs; the slot itself usually never is."""
    tree = session.tree
    index = session.scopes
    for fn in leaf_functions(tree.root):
        tracked = tracked_bindings(index, fn)
        if not tracked:
            continue
        multi = {n for n, b in tracked.items() if len(b.writes) > 1}
        if not multi:
            continue
        live = _Liveness(tracked)
        live.block(fn.get('body').children_in('body'), set())
        if live.unsafe:
            continue
        taken = used_names(fn)
        made = []
        for parent, slot in body_lists(fn):
            if not parent.live:
                continue
            items = list(parent.children_in(slot))
            for i, s in enumerate(items):
                d = assignment_of(s)
                if not d or d[0] not in multi:
                    continue
                name = d[0]
                j = None
                for t in range(i + 1, len(items)):
                    r, w = [], []
                    scan(items[t], r, w)
                    if any(x.attr('name') == name for x in w):
                        break                                  # redefined first: not a use of this def
                    if any(x.attr('name') == name for x in r):
                        j = t
                        break
                if j is None:
                    continue
                use = items[j]
                if name in live.after.get(use.nid, tracked.keys()):
                    continue
                if any(_escapes(x) for x in items[i + 1:j + 1]):
                    continue
                new = fresh('w', taken)
                made.append(new)
                tree.replace(d[1].get('left'), ident(new))
                reads, writes = [], []
                scan(use, reads, writes)
                for x in reads:
                    if x.attr('name') == name:
                        tree.replace(x, ident(new))
        declare(tree, fn, made)


def propagate_copies(session):
    """`x = y` makes reads of x into reads of y until either is written again.
    y must be written at most once: its value cannot change under the copy."""
    tree = session.tree
    index = session.scopes
    for fn in leaf_functions(tree.root):
        tracked = tracked_bindings(index, fn)
        if not tracked:
            continue
        sources = {n for n, b in tracked.items() if len(b.writes) <= 1}
        _Copies(tree, tracked, sources).block(fn.get('body'), {})


class _Copies:
    def __init__(self, tree, tracked, sources):
        self.tree, self.tracked, self.sources = tree, tracked, sources

    @staticmethod
    def kill(avail, names):
        for k in [k for k, v in avail.items() if k in names or v in names]:
            del avail[k]

    def rewrite(self, node, avail):
        if not avail:
            return
        reads, writes = [], []
        if node.kind == 'Identifier':
            reads = [node]
        else:
            scan(node, reads, writes)
        for r in reads:
            src = avail.get(r.attr('name'))
            if src is not None and r.live:
                self.tree.replace(r, ident(src))

    def block(self, b, avail):
        for s in list(b.children_in('body')):
            if s.live:
                self.stmt(s, avail)

    def sub(self, n, avail):
        if n is None:
            return
        if n.kind == 'BlockStatement':
            self.block(n, avail)
        else:
            self.stmt(n, avail)

    def stmt(self, s, avail):
        k = s.kind
        if k == 'ExpressionStatement':
            d = assignment_of(s)
            self.rewrite(s.get('expression') if not d else d[1].get('right'), avail)
            if not d:
                if names_written(s):
                    self.kill(avail, names_written(s))
                return
            name, assign = d
            self.kill(avail, {name})
            right = assign.get('right')
            if name in self.tracked and right.kind == 'Identifier' and right.attr('name') in self.sources \
                    and right.attr('name') != name:
                avail[name] = right.attr('name')
            return
        if k in ('ReturnStatement', 'ThrowStatement'):
            if s.get('argument') is not None:
                self.rewrite(s.get('argument'), avail)
            return
        written = names_written(s) if k != 'IfStatement' else None
        if k == 'IfStatement':
            self.rewrite(s.get('test'), avail)
            before = dict(avail)
            for arm in (s.get('consequent'), s.get('alternate')):
                self.sub(arm, dict(before))
            self.kill(avail, names_written(s))
            return
        if k in ('WhileStatement', 'DoWhileStatement'):
            self.kill(avail, written)
            inner = dict(avail)
            self.sub(s.get('body'), inner)
            if k == 'WhileStatement':
                self.rewrite(s.get('test'), dict(avail))
            return
        if k == 'LabeledStatement':
            self.kill(avail, written)
            self.sub(s.get('body'), dict(avail))
            return
        if k == 'BlockStatement':
            self.block(s, avail)
            return
        if k == 'TryStatement':
            self.sub(s.get('block'), dict(avail))
            handler = s.get('handler')
            if handler is not None:
                self.sub(handler.get('body'), dict(avail))
            finalizer = s.get('finalizer')
            if finalizer is not None:
                self.sub(finalizer, dict(avail))
            self.kill(avail, names_written(s))
            return
        self.kill(avail, written or set())


def fold_constants(session):
    """Replace what CCP proves constant with literals, and collapse branches
    whose test it can decide. CCP already knows `s1 = 3; ...; s0[s1]` holds on
    every path, including through joins that straight-line propagation misses."""
    tree = session.tree
    index = session.scopes
    facts_by_fn = session.facts
    edits, branches = [], []
    for fn in leaf_functions(tree.root):
        entry = facts_by_fn.get(fn.nid)
        tracked = tracked_bindings(index, fn)
        if entry is None or tracked is None:
            continue
        facts = entry[0]
        if any(n.kind in ('SwitchStatement', 'TryStatement') for n in walk(fn.get('body'))):
            continue      # an unstructured dispatcher (or a try): CCP's CFG does not model the implicit
                          # jumps (pc = vm_h in a catch), so a fold here can be wrong
        read_nodes = {id(r) for b in tracked.values() for r in b.reads}
        for n in walk(fn.get('body')):
            k = n.kind
            if k == 'Identifier':
                if id(n) not in read_nodes or n.attr('name') == 'undefined':
                    continue
            elif k in ('BinaryExpression', 'UnaryExpression', 'LogicalExpression', 'ConditionalExpression'):
                pass
            elif k == 'IfStatement':
                truth = facts.truth_of(n.get('test'))
                if truth is not V.UNKNOWN and facts.value_of(n.get('test')) not in V.LATTICE_MARKERS:
                    branches.append((n, bool(truth)))
                continue
            else:
                continue
            value = facts.constant_of(n)
            if value is None:
                continue
            lit = literal_node(value)
            if lit is not None:
                edits.append((n, lit))
    for target, lit in edits:
        if target.live and target.parent is not None:
            tree.replace(target, lit)
    for stmt, truth in branches:
        if not stmt.live or stmt.parent is None:
            continue
        arm = stmt.get('consequent') if truth else stmt.get('alternate')
        body = [] if arm is None else (arm.children_in('body') if arm.kind == 'BlockStatement' else [arm])
        if isinstance(stmt.parent.children.get(stmt.parent_slot), list):
            tree.replace_many(stmt, [clone(x) for x in body])
        else:
            tree.replace(stmt, make('BlockStatement', body=[clone(x) for x in body]))


def remove_dead_stores(session):
    tree = session.tree
    index = session.scopes
    for fn in leaf_functions(tree.root):
        tracked = tracked_bindings(index, fn)
        if not tracked:
            continue
        for parent, slot in body_lists(fn):
            for s in list(parent.children_in(slot)):
                d = assignment_of(s) if s.live else None
                if not d or d[0] not in tracked or tracked[d[0]].reads:
                    continue
                right = d[1].get('right')
                if pure(right):
                    tree.remove(s)
                else:
                    tree.replace(s, make('ExpressionStatement', expression=clone(right)))


def prune_declarations(session):
    tree = session.tree
    index = session.scopes
    for fn in leaf_functions(tree.root):
        tracked = tracked_bindings(index, fn)
        if tracked is None:
            continue
        for decl in [d for d in fn.get('body').children_in('body') if d.kind == 'VariableDeclaration']:
            keep = []
            for d in decl.children_in('declarations'):
                b = tracked.get(d.get('id').attr('name'))
                dead = b is not None and d.get('init') is None and not b.reads and not b.writes
                if not dead:
                    keep.append(d)
            if len(keep) == len(decl.children_in('declarations')):
                continue
            if keep:
                tree.set_slot(decl, 'declarations', keep)
            else:
                tree.remove(decl)


def fold_declarations(session):
    """`var x; ... x = E;`  ->  `var x = E;` at the first top-level assignment.
    `var` hoists, so moving the declaration to where the value first appears
    changes nothing; it just reads the way it was written. Last, because a
    declarator initialiser counts as a write and hides single-write temporaries
    from the other passes."""
    tree = session.tree
    for fn in [n for n in walk(tree.root, kinds=FUNCTION_KINDS)]:
        body = fn.get('body')
        if body is None or body.kind != 'BlockStatement':
            continue
        items = list(body.children_in('body'))
        bare = {}
        for st in items:
            if st.kind == 'VariableDeclaration' and st.attr('kind') == 'var':
                for d in st.children_in('declarations'):
                    if d.get('init') is None:
                        bare[d.get('id').attr('name')] = st
        for st in items:
            a = assignment_of(st) if st.live else None
            if not a or a[0] not in bare:
                continue
            holder = bare.pop(a[0])
            tree.replace(st, make('VariableDeclaration', kind='var', declarations=[
                make('VariableDeclarator', id=ident(a[0]), init=clone(a[1].get('right')))]))
            rest = [d for d in holder.children_in('declarations') if d.get('id').attr('name') != a[0]]
            if rest:
                tree.set_slot(holder, 'declarations', rest)
            else:
                tree.remove(holder)
