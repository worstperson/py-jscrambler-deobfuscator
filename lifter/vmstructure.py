"""
vmstructure -- restore if/else and loops from a lifted block graph.

vmlift emits dispatcher form (`for (;;) switch (pc) { case n: ... }`). This
module turns the block graph behind that dispatcher back into structured
statements, using the analyses in jsir.flow:

  * dominators / natural_loops   -- which blocks are loop headers, and what is in
                                    each loop (flow.Dominators.natural_loops)
  * dominators on the REVERSED   -- the join point of a conditional is the
    graph                           nearest common post-dominator of its arms

Shape of the algorithm
----------------------
Regions.  The graph is analysed one region at a time: the top level, then the
body of each loop, with every nested loop contracted to its header. Inside a
loop body an edge back to the header (`continue`) or out of the loop (`break`)
is *abrupt*: it ends the arm and takes no part in deciding where the arms
rejoin. That is what lets an `if / else if` chain whose arms all end in
`continue`, plus one arm that `return`s, still come out as a chain.

Joins.  For a branch, the join is the nearest common post-dominator of its flow
arms in the region graph. One flow arm: that arm is the join (the other arm
ends abruptly, so it becomes `if (c) { ...; break; }` followed by the rest). No
flow arms: there is no join, both arms are emitted to their end.

Loops.  Each natural loop gets one continuation block (the single non-terminal
exit, or the header's exit edge when every exit is terminal). Exit targets that
only lead to `return` / `throw` are emitted inline where they are reached.
Exits to an enclosing loop's header/continuation become labelled
`continue` / `break`. Loop shapes recognised afterwards:

    for(;;) { S; if (c) continue; else break; }   ->  do { S } while (c)
    for(;;) { if (c) break; S }                   ->  while (!c) { S }
    otherwise                                     ->  while (true) { ... }

What it will not do (raises Unstructurable; the caller keeps the dispatcher):
irreducible control flow, exception handlers, a loop with two non-terminal exits,
and any block that would have to be emitted twice (needs a goto or tail
duplication). Small terminal blocks (`return` / `throw`) are duplicated freely.

Condition tidying.  A single-use `vN = !x` / `vN = a === b` temporary defined in
the branching block is flipped in place when that makes the branch read better
(`if (!v) A else B` -> `if (v') B else A`), instead of stacking a `!` on top.

Not done here (separate, later passes): folding the vN temporaries back into
expressions (so `while (true) { v35 = ...; if (v43) break; }` becomes
`while (s[3] !== ...)`), and turning `s0 = a / s0 = b` diamonds into `?:`.

Usage
    python vmstructure.py LIFTED.js [-o OUT.js]      restructure dispatcher-form JS
    from vmstructure import structured_body           inside vmlift (see patch)
"""

import sys

from jsir import node as make, clone, walk
from jsir.flow import dominators

from lifter.vmgraph import Unstructurable, STerm, SBlock, Graph

END = -1                       # virtual exit of a region graph (never a block id)
DUP_LIMIT = 12                 # a terminal block up to this size may be emitted more than once
EXIT_KINDS = ('ReturnStatement', 'ThrowStatement')
LOOP_STMT_KINDS = ('WhileStatement', 'DoWhileStatement', 'ForStatement',
                   'ForInStatement', 'ForOfStatement')
FLIP = {'===': '!==', '!==': '===', '==': '!=', '!=': '=='}


# Unstructurable, STerm, SBlock and Graph live in vmgraph (shared with vmexc).


# ---------------------------------------------------------------------------
# The graph the structurer works on
# ---------------------------------------------------------------------------

def graph_from_lifted(lifted, ids):
    """vmlift.Lifted -> Graph. `ids` maps block state -> case id."""
    blocks = {}
    for lb in lifted.blocks:
        t = lb.term
        if t.kind == 'goto':
            term = STerm('goto', target=ids[t.target])
        elif t.kind == 'branch':
            term = STerm('branch', cond=t.cond, t=ids[t.t], f=ids[t.f])
        else:
            term = STerm(t.kind, value=t.value)
        blocks[lb.id] = SBlock(lb.id, list(lb.stmts), term)
    return Graph(0, blocks, has_exc=bool(lifted.has_exc))


# ---------------------------------------------------------------------------
# Small node helpers
# ---------------------------------------------------------------------------

def _ident(name):
    return make('Identifier', name=name)


def _block(stmts):
    return make('BlockStatement', body=list(stmts))


def _not(expr):
    return make('UnaryExpression', operator='!', prefix=True, argument=expr)


def _negate_expr(expr):
    """The logical negation of an expression, without stacking `!!`."""
    if expr.kind == 'UnaryExpression' and expr.attr('operator') == '!':
        return clone(expr.get('argument'))
    if expr.kind == 'BinaryExpression' and expr.attr('operator') in FLIP:
        return make('BinaryExpression', operator=FLIP[expr.attr('operator')],
                    left=clone(expr.get('left')), right=clone(expr.get('right')))
    return _not(clone(expr))


def _body_of(n):
    if n is None:
        return []
    if n.kind == 'BlockStatement':
        return list(n.children_in('body'))
    return [n]


def _make_if(test, then, other):
    """Build an if from statement lists, normalising empty arms. Returns a list
    of statements (empty when both arms are empty: the test is a pure atom)."""
    then, other = list(then), list(other or [])
    if not then and not other:
        return []
    if not then:
        test, then, other = _negate_expr(test), other, []
    alternate = None
    if len(other) == 1 and other[0].kind == 'IfStatement':
        alternate = other[0]                    # else if
    elif other:
        alternate = _block(other)
    return [make('IfStatement', test=test, consequent=_block(then), alternate=alternate)]


def _is_plain(n, kind):
    return n.kind == kind and n.get('label') is None


def _ends_in_exit(stmts):
    return bool(stmts) and stmts[-1].kind in EXIT_KINDS


def _size(stmts):
    return sum(1 for s in stmts for n in walk(s) if n.is_statement)


def _assign_target(stmt):
    """`name = ...;` -> (name, assignment node), else None."""
    if stmt.kind != 'ExpressionStatement':
        return None
    e = stmt.get('expression')
    if e.kind == 'AssignmentExpression' and e.attr('operator') == '=' \
            and e.get('left').kind == 'Identifier':
        return e.get('left').attr('name'), e
    return None


def _thread_empty_blocks(graph):
    """Jump threading: a block with no statements that only jumps elsewhere is
    transparent, so edges to it are redirected to where it goes. Without this a
    trivial `goto` block shared by two arms looks like a join that needs
    duplicating. Returns a new block table; the graph passed in is not touched."""
    blocks = graph.blocks

    def final(b):
        seen = set()
        while b in blocks and b not in seen and not blocks[b].stmts \
                and blocks[b].term.kind == 'goto':
            seen.add(b)
            b = blocks[b].term.target
        return b

    out = {}
    for bid, blk in blocks.items():
        t = blk.term
        if t.kind == 'goto':
            term = STerm('goto', target=final(t.target))
        elif t.kind == 'branch':
            term = STerm('branch', cond=t.cond, t=final(t.t), f=final(t.f))
        else:
            term = t
        out[bid] = SBlock(bid, blk.stmts, term, blk.tag)
    return out


# ---------------------------------------------------------------------------
# Loop bookkeeping
# ---------------------------------------------------------------------------

class Loop:
    def __init__(self, header, nodes, latches):
        self.header, self.nodes, self.latches = header, nodes, latches
        self.parent = None
        self.children = []
        self.cont = None            # block control continues at after the loop, or None
        self.label = None

    def __repr__(self):
        return f'<Loop {self.header} n={len(self.nodes)} cont={self.cont}>'


class _Region:
    """The flow graph of one region (a loop body or the top level), with its
    post-dominator relation."""

    def __init__(self, members, flow, ipdom):
        self.members, self.flow, self.ipdom = members, flow, ipdom


# ---------------------------------------------------------------------------
# The structurer
# ---------------------------------------------------------------------------

class Structurer:
    def __init__(self, graph):
        if graph.has_exc:
            from lifter import vmexc
            graph = vmexc.prepare(graph, lambda g: Structurer(g).run(), _thread_empty_blocks)
        self.graph = graph
        self.blocks = _thread_empty_blocks(graph)
        self.entry = graph.entry
        self.emitted = set()
        self._regions = {}
        self._labels = 0
        self._analyse()

    # -- analysis -------------------------------------------------------------

    def _analyse(self):
        blocks = self.blocks
        if self.entry not in blocks:
            raise Unstructurable('no entry block')

        succ = {}
        for bid, blk in blocks.items():
            t = blk.term
            if t.kind == 'goto':
                out = [t.target]
            elif t.kind == 'branch':
                out = [t.t] if t.t == t.f else [t.t, t.f]
            elif t.kind in ('return', 'throw', 'fall'):
                out = []
            else:
                raise Unstructurable(f'block {bid}: unknown terminator {t.kind}')
            for target in out:
                if target not in blocks:
                    raise Unstructurable(f'block {bid} jumps to missing block {target}')
            succ[bid] = out

        reach, stack = set(), [self.entry]
        while stack:
            b = stack.pop()
            if b not in reach:
                reach.add(b)
                stack.extend(succ[b])
        self.reach = reach
        self.succ = {b: succ[b] for b in reach}
        self.pred = {b: [] for b in reach}
        for b in reach:
            for s in self.succ[b]:
                self.pred[s].append(b)

        doms = dominators(None, self.entry,
                          successors=self.succ.__getitem__,
                          predecessors=self.pred.__getitem__)
        self.doms = doms
        rank = {b: i for i, b in enumerate(doms.order)}
        for u in reach:                                   # reducibility
            for v in self.succ[u]:
                if rank[v] <= rank[u] and not doms.dominates(v, u):
                    raise Unstructurable(f'irreducible control flow ({u} -> {v})')

        found = doms.natural_loops(doms.order, self.succ.__getitem__, self.pred.__getitem__)
        self.loops = {h: Loop(h, info['nodes'], info['latches']) for h, info in found.items()}
        ordered = sorted(self.loops.values(), key=lambda L: len(L.nodes))
        for L in ordered:                                 # nest
            for M in ordered:
                if M is not L and len(M.nodes) > len(L.nodes) and L.header in M.nodes:
                    L.parent = M
                    break
        for L in ordered:
            if L.parent is not None:
                L.parent.children.append(L)
        self.owner = {}                                   # innermost loop containing a block
        for L in ordered:
            for n in L.nodes:
                self.owner.setdefault(n, L)

        for L in sorted(self.loops.values(), key=lambda L: -len(L.nodes)):
            self._choose_continuation(L)

        self.uses = {}                                    # identifier name -> occurrences
        for b in reach:
            blk = self.blocks[b]
            parts = list(blk.stmts)
            if blk.term.cond is not None:
                parts.append(blk.term.cond)
            if blk.term.value is not None:
                parts.append(blk.term.value)
            for part in parts:
                for n in walk(part, kinds={'Identifier'}):
                    name = n.attr('name')
                    self.uses[name] = self.uses.get(name, 0) + 1

    def _reach_from(self, start):
        seen, stack = set(), [start]
        while stack:
            b = stack.pop()
            if b not in seen:
                seen.add(b)
                stack.extend(self.succ[b])
        return seen

    def _choose_continuation(self, L):
        ancestors = []
        a = L.parent
        while a is not None:
            ancestors.append(a)
            a = a.parent
        abrupt = set()
        for a in ancestors:
            abrupt.add(a.header)
            if a.cont is not None:
                abrupt.add(a.cont)
        exits = []
        for u in sorted(L.nodes):
            for v in self.succ[u]:
                if v not in L.nodes and v not in exits:
                    exits.append(v)
        candidates = [v for v in exits if v not in abrupt]
        reaches = {v: self._reach_from(v) for v in candidates}

        def private(v):
            s = reaches[v]
            if s & L.nodes or any(b in self.loops for b in s):
                return False
            if any(self.blocks[b].term.kind == 'fall' for b in s):
                return False                       # leaving a nested region is a real exit, never inlined
            if not all(self.doms.dominates(L.header, b) for b in s):
                return False                   # also reachable without going through the loop
            return all(not (s & reaches[w]) for w in candidates if w != v)

        regular = [v for v in candidates if not private(v)]
        if len(regular) > 1:
            # An arm that ends in `break`/`return` cannot reach the latch, so the
            # natural loop leaves it out and it looks like a second exit. If the
            # header dominates it and it leads to one common follow block, it is
            # body code, and the follow block is the continuation.
            follow = [f for f in regular
                      if all(f == v or (f in reaches[v] and self.doms.dominates(L.header, v))
                             for v in regular)]
            if not follow:
                raise Unstructurable(
                    f'loop at block {L.header} has {len(regular)} non-terminal exits {regular}')
            regular = follow[:1]
        if regular:
            L.cont = regular[0]
        else:
            from_header = [v for v in self.succ[L.header] if v in candidates]
            L.cont = from_header[0] if from_header else None

    # -- regions --------------------------------------------------------------

    def _child_loop(self, R, v):
        l, prev = self.owner.get(v), None
        while l is not None and l is not R:
            prev, l = l, l.parent
        return prev if l is R else None

    def _rep(self, R, v):
        """Where an edge to v lands in region R's graph (a nested loop is its header)."""
        c = self._child_loop(R, v)
        if c is None:
            return v
        if v != c.header:
            raise Unstructurable(f'edge enters the loop at {c.header} other than through its header')
        return c.header

    @staticmethod
    def _abrupt(R, x):
        return R is not None and (x == R.header or x not in R.nodes)

    def _region(self, R):
        key = R.header if R is not None else None
        cached = self._regions.get(key)
        if cached is not None:
            return cached
        members = set()
        for b in self.reach:
            if R is not None and b not in R.nodes:
                continue
            c = self._child_loop(R, b)
            if c is None or b == c.header:
                members.add(b)
        flow = {}
        for u in members:
            c = self.loops.get(u)
            if c is not None and c is not R:              # a contracted child loop
                targets = [] if c.cont is None else [self._rep(R, c.cont)]
            else:
                targets = [self._rep(R, v) for v in self.succ[u]]
            flow[u] = [x for x in targets if not self._abrupt(R, x) and x in members]
        rpred = {n: [] for n in members}
        for u, outs in flow.items():
            for x in outs:
                rpred[x].append(u)
        sinks = [u for u in members if not flow[u]]
        rsucc = dict(rpred)
        rsucc[END] = list(sinks)
        rpred_full = {u: list(flow[u]) + ([END] if u in sinks else []) for u in members}
        rpred_full[END] = []
        ipdom = dominators(None, END,
                           successors=rsucc.__getitem__,
                           predecessors=rpred_full.__getitem__)
        built = _Region(members, flow, ipdom)
        self._regions[key] = built
        return built

    def _nca(self, R, a, b):
        idom = self._region(R).ipdom.idom
        if a not in idom or b not in idom:
            return None
        seen, x = set(), a
        while True:
            seen.add(x)
            if idom[x] == x:
                break
            x = idom[x]
        y = b
        while y not in seen:
            if idom[y] == y:
                return None
            y = idom[y]
        return None if y == END else y

    def _first_common(self, R, a, b):
        """Earliest block both arms can reach in the region's flow graph.

        The post-dominator join is END as soon as one arm contains an early
        `return`, since that path reaches the exit without passing the code the
        other arms share. The earliest common block is the right join there.
        It is only a candidate: if some path bypasses it, emission reaches a
        block twice and the function is left in dispatcher form."""
        flow = self._region(R).flow

        def reach(start):
            seen, stack = set(), [start]
            while stack:
                x = stack.pop()
                if x not in seen:
                    seen.add(x)
                    stack.extend(flow.get(x, ()))
            return seen
        common = reach(a) & reach(b)
        if not common:
            return None
        rank = {n: i for i, n in enumerate(self.doms.order)}
        return min(common, key=lambda n: rank[n])

    # -- emission -------------------------------------------------------------

    def run(self):
        out = []
        self._seq(self.entry, None, [], out)
        missing = self.reach - self.emitted
        if missing:
            raise Unstructurable(f'blocks never emitted: {sorted(missing)}')
        return out

    def _label(self, L):
        if L.label is None:
            self._labels += 1
            L.label = f'L{self._labels}'
        return L.label

    def _jump(self, b, ctxs):
        for i in range(len(ctxs) - 1, -1, -1):
            L = ctxs[i]
            if b == L.header:
                kind = 'ContinueStatement'
            elif L.cont is not None and b == L.cont:
                kind = 'BreakStatement'
            else:
                continue
            label = None if i == len(ctxs) - 1 else _ident(self._label(L))
            return make(kind, label=label)
        return None

    def _seq(self, b, join, ctxs, out, entering=None):
        while b is not None:
            if b == join:
                return
            if b != entering:
                jump = self._jump(b, ctxs)
                if jump is not None:
                    out.append(jump)
                    return
                loop = self.loops.get(b)
                if loop is not None:
                    b = self._loop(loop, ctxs, out)
                    continue
            entering = None
            blk = self.blocks[b]
            if b in self.emitted and not (blk.term.kind in ('return', 'throw', 'fall')
                                          and len(blk.stmts) <= DUP_LIMIT):
                raise Unstructurable(f'block {b} is reached twice (would need a goto or tail duplication)')
            self.emitted.add(b)
            out.extend(clone(s) for s in blk.stmts)
            t = blk.term
            if t.kind == 'goto':
                b = t.target
            elif t.kind == 'fall':
                return                                     # the end of a nested region: continue after it
            elif t.kind == 'return':
                out.append(make('ReturnStatement', argument=clone(t.value) if t.value is not None else None))
                return
            elif t.kind == 'throw':
                out.append(make('ThrowStatement', argument=clone(t.value)))
                return
            else:
                b = self._branch(blk, join, ctxs, out)

    # -- branches -------------------------------------------------------------

    def _branch(self, blk, join, ctxs, out):
        t = blk.term
        R = self.owner.get(blk.bid)
        tx, fx = self._rep(R, t.t), self._rep(R, t.f)
        flow = [x for x in (tx, fx) if not self._abrupt(R, x)]
        if not flow:
            jn = None
        elif len(flow) == 1 or tx == fx:
            jn = flow[0]
        else:
            jn = self._nca(R, tx, fx)
            if jn is None:
                jn = self._first_common(R, tx, fx)

        # With no join of its own, an arm still stops at the enclosing one.
        stop = jn if jn is not None else join
        then_s, else_s = [], []
        self._seq(t.t, stop, ctxs, then_s)
        self._seq(t.f, stop, ctxs, else_s)
        nb = len(blk.stmts)
        cond = clone(t.cond)

        if not then_s and not else_s:
            return jn

        if then_s and else_s:
            if _ends_in_exit(then_s):                      # guard clause: no else needed
                out.extend(_make_if(cond, then_s, []))
                out.extend(else_s)
                return jn
            kind = self._free_negation(out, nb, cond)
            bigger = _size(then_s) - _size(else_s)
            if kind is not None and (bigger > 0 or (bigger == 0 and kind == 'not')):
                cond = self._negate(out, nb, cond)
                then_s, else_s = else_s, then_s
            out.extend(_make_if(cond, then_s, else_s))
        elif then_s:
            out.extend(_make_if(cond, then_s, []))
        else:
            out.extend(_make_if(self._negate(out, nb, cond), else_s, []))
        return jn

    def _definition(self, out, nb, cond):
        """(statement, assignment) of the `cond = ...;` inside this block's
        emitted statements, if cond is a temporary used nowhere but there and
        at the branch."""
        if cond.kind != 'Identifier' or self.uses.get(cond.attr('name'), 0) != 2:
            return None
        name = cond.attr('name')
        for stmt in reversed(out[max(0, len(out) - nb):]):
            hit = _assign_target(stmt)
            if hit is not None and hit[0] == name:
                return stmt, hit[1]
        return None

    def _free_negation(self, out, nb, cond):
        """'not' / 'cmp' when negating cond costs nothing (its single-use
        definition can absorb it), else None."""
        d = self._definition(out, nb, cond)
        if d is None:
            return None
        right = d[1].get('right')
        if right.kind == 'UnaryExpression' and right.attr('operator') == '!':
            return 'not'
        if right.kind == 'BinaryExpression' and right.attr('operator') in FLIP:
            return 'cmp'
        return None

    def _negate(self, out, nb, cond):
        if self._free_negation(out, nb, cond) is not None:
            stmt, d = self._definition(out, nb, cond)
            right = d.get('right')
            if right.kind == 'UnaryExpression':
                inner = right.get('argument')
                if inner.kind == 'Identifier':          # `v = !w` -> the branch tests w itself
                    out[:] = [o for o in out if o is not stmt]
                    return clone(inner)
                d.children['right'] = inner
                inner.parent, inner.parent_slot = d, 'right'
            else:
                right.attrs['operator'] = FLIP[right.attr('operator')]
            return cond
        return _negate_expr(cond)

    # -- loops ----------------------------------------------------------------

    def _loop(self, L, ctxs, out):
        body = []
        self._seq(L.header, None, ctxs + [L], body, entering=L.header)
        stmt = self._shape_loop(L, body)
        if L.label is not None:
            stmt = make('LabeledStatement', label=_ident(L.label), body=stmt)
        out.append(stmt)
        return L.cont

    def _shape_loop(self, L, body):
        # do { S } while (c)
        if body and body[-1].kind == 'IfStatement' and self._is_loop_test(body[-1]):
            last = body[-1]
            cons = _body_of(last.get('consequent'))[0]
            if not self._continues_to(body[:-1], L):
                test = clone(last.get('test'))
                if cons.kind != 'ContinueStatement':
                    test = _negate_expr(test)
                return make('DoWhileStatement', body=_block(body[:-1]), test=test)
        body = self._strip_tail_continue(body)
        # while (!c) { S }
        if body and body[0].kind == 'IfStatement' and body[0].get('alternate') is None:
            cons = _body_of(body[0].get('consequent'))
            if len(cons) == 1 and _is_plain(cons[0], 'BreakStatement'):
                return make('WhileStatement', test=_negate_expr(body[0].get('test')),
                            body=_block(body[1:]))
        return make('WhileStatement', test=make('Literal', value=True), body=_block(body))

    @staticmethod
    def _is_loop_test(n):
        """`if (c) continue; else break;` in either order, both unlabelled."""
        alt = n.get('alternate')
        if alt is None or alt.kind != 'BlockStatement':
            return False
        a, b = _body_of(n.get('consequent')), _body_of(alt)
        if len(a) != 1 or len(b) != 1:
            return False
        kinds = {a[0].kind, b[0].kind}
        return kinds == {'ContinueStatement', 'BreakStatement'} and \
            _is_plain(a[0], a[0].kind) and _is_plain(b[0], b[0].kind)

    def _continues_to(self, stmts, L):
        """Does any statement here continue loop L? (Matters for do-while: a
        `continue` there runs the test, in `for(;;)` it does not.)"""
        def visit(n, depth):
            if n.kind == 'ContinueStatement':
                label = n.get('label')
                if label is None:
                    return depth == 0
                return L.label is not None and label.attr('name') == L.label
            inner = depth + (1 if n.kind in LOOP_STMT_KINDS else 0)
            return any(visit(c, inner) for _s, c, _r in n.iter_children())
        return any(visit(s, 0) for s in stmts)

    def _strip_tail_continue(self, stmts):
        stmts = list(stmts)
        while stmts and _is_plain(stmts[-1], 'ContinueStatement'):
            stmts.pop()
        # `if (c) continue; rest` at the tail of the loop body  ->  `if (!c) { rest }`
        for i, s in enumerate(stmts):
            if s.kind == 'IfStatement' and s.get('alternate') is None:
                cons = _body_of(s.get('consequent'))
                if len(cons) == 1 and _is_plain(cons[0], 'ContinueStatement'):
                    rest = self._strip_tail_continue(stmts[i + 1:])
                    return stmts[:i] + _make_if(_negate_expr(s.get('test')), rest, [])
        if stmts and stmts[-1].kind == 'IfStatement':
            n = stmts.pop()
            then = self._strip_tail_continue(_body_of(n.get('consequent')))
            alt = n.get('alternate')
            other = self._strip_tail_continue(_body_of(alt)) if alt is not None else []
            stmts += _make_if(clone(n.get('test')), then, other)
        return stmts


def structure(graph):
    """Graph -> list of statements. Raises Unstructurable."""
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, 20000))
    try:
        return Structurer(graph).run()
    except RecursionError:
        raise Unstructurable('nesting too deep') from None
    finally:
        sys.setrecursionlimit(limit)


# ---------------------------------------------------------------------------
# Integration with vmlift
# ---------------------------------------------------------------------------

def _decl(names):
    return make('VariableDeclaration', kind='var',
                declarations=[make('VariableDeclarator', id=_ident(n), init=None) for n in names])


def structured_body(lifted, ids):
    """Drop-in for vmlift.dispatcher_body (production mode): declarations plus
    structured statements, with no `pc`. Raises Unstructurable."""
    stmts = structure(graph_from_lifted(lifted, ids))
    names = [f'v{i}' for i in range(lifted.temps)] + [f's{i}' for i in range(max(lifted.max_stack, 1 if lifted.has_exc else 0))]
    return ([_decl(names)] if names else []) + stmts


# ---------------------------------------------------------------------------
# Restructuring dispatcher-form JavaScript (an existing *_lifted.js)
# ---------------------------------------------------------------------------

def _pc_assign(stmt):
    hit = _assign_target(stmt)
    if hit is None or hit[0] != 'pc' or hit[1].get('right').kind != 'Literal':
        return None
    return int(hit[1].get('right').attr('value'))


def _is_budget_guard(n):
    """`if (++vm_steps > N) { throw ... }`: the test-mode runaway-loop bound."""
    t = n.get('test') if n.kind == 'IfStatement' else None
    return t is not None and t.kind == 'BinaryExpression' and t.attr('operator') == '>' \
        and t.get('left').kind == 'UpdateExpression' and t.get('left').get('argument').kind == 'Identifier' \
        and t.get('left').get('argument').attr('name') == 'vm_steps'


def _is_dispatch_catch(handler):
    """The lifter's catch clause:
        if (vm_h < 0) throw e;  s0 = e;  pc = vm_h;  vm_h = -1;  continue;"""
    if handler is None or handler.get('param') is None:
        return False
    e = handler.get('param').attr('name')
    body = _body_of(handler.get('body'))
    if len(body) != 5:
        return False

    def assign(st, left, right):
        a = st.get('expression') if st.kind == 'ExpressionStatement' else None
        return a is not None and a.kind == 'AssignmentExpression' and a.get('left').kind == 'Identifier' \
            and a.get('left').attr('name') == left and (right is None or (
                a.get('right').kind == 'Identifier' and a.get('right').attr('name') == right))
    guard = body[0]
    return (guard.kind == 'IfStatement' and guard.get('alternate') is None
            and guard.get('test').kind == 'BinaryExpression' and guard.get('test').attr('operator') == '<'
            and any(n.kind == 'ThrowStatement' for n in walk(guard.get('consequent'))))\
        and assign(body[1], 's0', e) and assign(body[2], 'pc', 'vm_h') and assign(body[3], 'vm_h', None) \
        and body[4].kind == 'ContinueStatement'


def graph_from_dispatcher(loop):
    """A `for (;;) { [guard] [try] switch (pc) { ... } [catch] }` statement -> Graph."""
    body = loop.get('body')
    inner = list(_body_of(body))
    if inner and _is_budget_guard(inner[0]):
        inner = inner[1:]                                   # test-mode only; meaningless once the loops are real
    if len(inner) != 1:
        raise Unstructurable('dispatcher loop has extra statements')
    switch, has_exc = inner[0], False
    if switch.kind == 'TryStatement':
        items = _body_of(switch.get('block'))
        if switch.get('finalizer') is not None or len(items) != 1 or not _is_dispatch_catch(switch.get('handler')):
            raise Unstructurable('unrecognised try around the dispatcher')
        switch, has_exc = items[0], True
    if switch.kind != 'SwitchStatement' or switch.get('discriminant').kind != 'Identifier' \
            or switch.get('discriminant').attr('name') != 'pc':
        raise Unstructurable('not a `switch (pc)` dispatcher')
    blocks = {}
    for case in switch.children_in('cases'):
        test = case.get('test')
        if test is None:
            continue
        bid = int(test.attr('value'))
        stmts = list(case.children_in('consequent'))
        if not stmts:
            raise Unstructurable(f'case {bid} is empty')
        last = stmts[-1]
        if last.kind == 'ReturnStatement':
            term, stmts = STerm('return', value=last.get('argument')), stmts[:-1]
        elif last.kind == 'ThrowStatement':
            term, stmts = STerm('throw', value=last.get('argument')), stmts[:-1]
        elif last.kind == 'ContinueStatement' and len(stmts) >= 2:
            prev = stmts[-2]
            target = _pc_assign(prev) if prev.kind == 'ExpressionStatement' else None
            if target is not None:
                term, stmts = STerm('goto', target=target), stmts[:-2]
            elif prev.kind == 'IfStatement' and prev.get('alternate') is not None:
                a, b = _body_of(prev.get('consequent')), _body_of(prev.get('alternate'))
                ta = _pc_assign(a[0]) if len(a) == 1 else None
                tb = _pc_assign(b[0]) if len(b) == 1 else None
                if ta is None or tb is None:
                    raise Unstructurable(f'case {bid}: branch arms are not plain `pc = n`')
                term, stmts = STerm('branch', cond=prev.get('test'), t=ta, f=tb), stmts[:-2]
            else:
                raise Unstructurable(f'case {bid}: unrecognised terminator')
        else:
            raise Unstructurable(f'case {bid}: unrecognised terminator')
        blocks[bid] = SBlock(bid, stmts, term)
    return Graph(0, blocks, has_exc=has_exc)


def restructure_tree(tree):
    """Replace every dispatcher loop in `tree` with structured statements.
    Returns [(function_name, 'ok' | reason)]."""
    report = []
    for loop in [n for n in walk(tree.root, kinds={'ForStatement'})
                 if n.get('init') is None and n.get('test') is None and n.get('update') is None]:
        fn = loop.enclosing('FunctionExpression', 'FunctionDeclaration')
        name = fn.get('id').attr('name') if fn is not None and fn.get('id') is not None else '?'
        try:
            stmts = structure(graph_from_dispatcher(loop))
        except Unstructurable as why:
            report.append((name, str(why)))
            continue
        scope = loop.parent
        tree.replace_many(loop, stmts)
        for decl in list(scope.children_in('body')):
            if decl.kind == 'VariableDeclaration':
                rest = [d for d in decl.children_in('declarations')
                        if d.get('id').attr('name') not in ('pc', 'vm_h', 'vm_steps')]
                if len(rest) != len(decl.children_in('declarations')):
                    if rest:
                        tree.set_slot(decl, 'declarations', rest)
                    else:
                        tree.remove(decl)
        report.append((name, 'ok'))
    return report


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    from jsir import Session
    from lifter import vmlift
    session = Session.parse(open(argv[1]).read())
    report = restructure_tree(session.tree)
    text = vmlift.to_js(session.tree.root)
    if '-o' in argv:
        open(argv[argv.index('-o') + 1], 'w').write(text + '\n')
    else:
        print(text)
    for name, status in report:
        print(f'// {name}: {status}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
