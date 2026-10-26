"""
vmexc -- try/catch from the lifter's exception bookkeeping.

The lifter does not keep try blocks. It keeps a handler register: before an
instruction covered by a handler it writes `vm_h = <handler block>`, and the
dispatcher's catch clause jumps there (`pc = vm_h`, exception in s0). So a
protected region starts and stops in the middle of blocks:

    vm_h = (-1);  ...                      uncovered
    vm_h = 1;  v21 = vm_locals[1];  v22 = v21[8];  ...      covered by block 1
    vm_h = (-1);  pc = 3;

Rebuilding `try { ... } catch (e) { ... }` from that takes four steps.

split_handlers      drop the `vm_h` writes and cut every block where the covering
                    handler changes; each piece gets a tag (the handler, or -1).
                    A stretch of statements that cannot throw between two covered
                    stretches of the same handler is absorbed, since the lifter
                    leaves moves like `vm_locals[6] = v24` uncovered even when
                    they sit inside the source's try.
contract_try_regions
                    for a handler: its tagged pieces must have one entry; the
                    join is the immediate post-dominator of that entry once each
                    covered piece also has an edge to the handler (the point
                    where "the try finished" and "the catch finished" meet); the
                    body is what the entry reaches before the join, the catch is
                    what the handler reaches before it. Both may leave only
                    through the join. They are structured as nested graphs, and
                    the region becomes one block holding the TryStatement.
                    Innermost regions first, so nesting works.

Anything else raises Unstructurable and the function stays in dispatcher form:
several entries into a try, code leaving a try to anywhere but the join (a
`continue` out of it, say), a handler also reachable normally, uncovered code
that can throw inside the region, and `finally` (the lifter does not emit one).
"""

import re

from jsir import node as make, clone, walk
from jsir.flow import dominators

from lifter.vmgraph import Unstructurable, STerm, SBlock, Graph

END = -1
_ATOM = re.compile(r'^([vs]\d+|vm_[a-z_]+|undefined)$')


def _ident(name):
    return make('Identifier', name=name)


def _num(e):
    if e.kind == 'Literal' and isinstance(e.attr('value'), float):
        return e.attr('value')
    if e.kind == 'UnaryExpression' and e.attr('operator') == '-' and e.get('argument').kind == 'Literal' \
            and isinstance(e.get('argument').attr('value'), float):
        return -e.get('argument').attr('value')
    return None


def handler_assignment(stmt):
    """`vm_h = <number>;` -> the number, else None."""
    if stmt.kind != 'ExpressionStatement':
        return None
    e = stmt.get('expression')
    if e.kind == 'AssignmentExpression' and e.attr('operator', '=') == '=' and e.get('left').kind == 'Identifier' \
            and e.get('left').attr('name') == 'vm_h':
        v = _num(e.get('right'))
        return None if v is None else int(v)
    return None


def _nothrow_stmt(st):
    """A move between locals or into the locals array: cannot throw."""
    if st.kind != 'ExpressionStatement':
        return False
    e = st.get('expression')
    if e.kind != 'AssignmentExpression' or e.attr('operator', '=') != '=':
        return False
    right, left = e.get('right'), e.get('left')
    if not (right.kind == 'Literal' or (right.kind == 'Identifier' and _ATOM.match(right.attr('name')))):
        return False
    if left.kind == 'Identifier':
        return True
    return left.kind == 'MemberExpression' and left.get('object').kind == 'Identifier' \
        and left.get('object').attr('name') == 'vm_locals'


def _nothrow_block(blk):
    return blk.term.kind != 'throw' and all(_nothrow_stmt(s) for s in blk.stmts)


# ---------------------------------------------------------------------------
# 1. cut blocks at handler changes
# ---------------------------------------------------------------------------

def split_handlers(graph):
    blocks = {}
    fresh = max(graph.blocks) + 1
    for bid in sorted(graph.blocks):
        blk = graph.blocks[bid]
        segs, cur = [], -1                                   # [tag, statements]
        for st in blk.stmts:
            h = handler_assignment(st)
            if h is not None:
                cur = h
            elif segs and segs[-1][0] == cur:
                segs[-1][1].append(st)
            else:
                segs.append([cur, [st]])
        if blk.term.kind == 'throw' and cur >= 0 and (not segs or segs[-1][0] != cur):
            segs.append([cur, []])                           # the throw itself is covered
        merged = True
        while merged:                                        # absorb non-throwing gaps inside one handler's stretch
            merged = False
            for i in range(1, len(segs) - 1):
                if segs[i][0] < 0 and 0 <= segs[i - 1][0] == segs[i + 1][0] \
                        and all(_nothrow_stmt(x) for x in segs[i][1]):
                    segs[i - 1][1] += segs[i][1] + segs[i + 1][1]
                    del segs[i:i + 2]
                    merged = True
                    break
        if not segs:
            blocks[bid] = SBlock(bid, [], blk.term, -1)
            continue
        ids = [bid]
        for _ in segs[1:]:
            ids.append(fresh)
            fresh += 1
        for k, (tag, stmts) in enumerate(segs):
            last = k == len(segs) - 1
            term = blk.term if last else STerm('goto', target=ids[k + 1])
            if not stmts and not (last and blk.term.kind == 'throw'):
                tag = -1
            blocks[ids[k]] = SBlock(ids[k], stmts, term, tag)
    return Graph(graph.entry, blocks, True)


# ---------------------------------------------------------------------------
# 2. contract try/catch regions
# ---------------------------------------------------------------------------

def _reach(start, succ, stop=()):
    seen, stack = set(), [start]
    while stack:
        b = stack.pop()
        if b in seen or b in stop:
            continue
        seen.add(b)
        stack.extend(succ.get(b, ()))
    return seen


def _post_idom(nodes, succ):
    sinks = [n for n in nodes if not succ[n]]
    rsucc = {n: [] for n in nodes}
    rsucc[END] = list(sinks)
    for n in nodes:
        for s in succ[n]:
            rsucc[s].append(n)
    rpred = {n: list(succ[n]) + ([END] if not succ[n] else []) for n in nodes}
    rpred[END] = []
    return dominators(None, END, successors=rsucc.__getitem__, predecessors=rpred.__getitem__).idom


def _live(blocks, entry):
    """Drop code nothing can reach. A handler counts as reached from the code it covers."""
    seen, stack = set(), [entry]
    while stack:
        b = stack.pop()
        if b in seen or b not in blocks:
            continue
        seen.add(b)
        stack.extend(blocks[b].term.successors())
        if blocks[b].tag >= 0:
            stack.append(blocks[b].tag)
    return {b: blk for b, blk in blocks.items() if b in seen}


def contract_try_regions(graph, structure):
    """graph with tags -> graph without them. `structure(Graph) -> [statements]`
    structures a nested region (vmstructure passes its own structurer in)."""
    blocks = dict(graph.blocks)
    entry = graph.entry
    for _ in range(500):
        blocks = _live(blocks, entry)
        succ = {b: blk.term.successors() for b, blk in blocks.items()}
        reach = _reach(entry, succ)
        tags = sorted({blk.tag for blk in blocks.values() if blk.tag >= 0})
        if not tags:
            return Graph(entry, blocks, False)
        # A handler whose block is covered by another handler sits inside that one's try: inner first.
        ready = [h for h in tags if not any(blocks[x].tag == h for x in tags if x != h and x in blocks)]
        for h in ready:
            done = _contract_one(h, blocks, entry, succ, reach, structure)
            if done is not None:
                blocks = done
                break
        else:
            raise Unstructurable('try regions nest in a way that cannot be ordered')
    raise Unstructurable('too many try regions')


def _contract_one(h, blocks, entry, succ, reach, structure):
    """Contract the region of handler h, or return None when it still contains
    another handler's code (do that one first)."""
    if h not in blocks:
        raise Unstructurable(f'handler block {h} does not exist')
    covered = [b for b in reach if blocks[b].tag == h]
    if not covered:
        return None                                           # only reachable through another handler: that goes first
    pred = {b: [] for b in reach}
    for b in reach:
        for s in succ[b]:
            pred[s].append(b)
    doms = dominators(None, entry, successors=lambda b: succ[b], predecessors=lambda b: pred[b])
    top = next((c for c in doms.order if c in covered and all(doms.dominates(c, t) for t in covered)), None)
    if top is None:
        raise Unstructurable(f'the code covered by handler {h} has several entry points')

    # The try body: the covered code, plus whatever lies on a path from its entry to covered code.
    reach_top = _reach(top, succ)
    feeds, stack = set(covered), list(covered)
    while stack:
        n = stack.pop()
        for p in pred.get(n, ()):
            if p in reach_top and p not in feeds:
                feeds.add(p)
                stack.append(p)
    body = feeds
    if h in body:
        raise Unstructurable(f'handler {h} is reachable from inside its own try')
    if any(blocks[n].tag >= 0 and blocks[n].tag != h for n in body):
        return None                                           # another handler's code is inside: it goes first
    for n in body:
        if blocks[n].tag != h and not _nothrow_block(blocks[n]):
            raise Unstructurable(f'uncovered code that can throw inside the try of handler {h}')

    # The join: where the try body leaves to, and where the catch must rejoin.
    exits = sorted({s for n in body for s in succ[n] if s not in body})
    if len(exits) > 1:
        raise Unstructurable(f'the try of handler {h} leaves to several places {exits}')
    if exits:
        join = exits[0]
        after = _reach(join, succ)
        if h in after:
            raise Unstructurable(f'handler {h} is also reachable after its try')
        handler = _reach(h, succ, after)
        hits = {s for n in handler for s in succ[n] if s in after}
        if len(hits) > 1:
            raise Unstructurable(f'the catch of handler {h} rejoins at several places {sorted(hits)}')
        if hits and hits != {join}:
            meet = next(iter(hits))                           # the try's exit is a short non-throwing tail
            cur, chain = join, []
            while cur != meet:
                node = blocks[cur]
                if cur in handler or cur in chain or node.tag >= 0 or not _nothrow_block(node) \
                        or len(succ[cur]) != 1:
                    raise Unstructurable(f'the try and catch of handler {h} rejoin at different places')
                chain.append(cur)
                cur = succ[cur][0]
            body = body | set(chain)
            join = meet
            handler = _reach(h, succ, {join})
    else:
        # The body never completes normally; what follows is what the handler continues to.
        universe = reach | _reach(h, succ)
        if any(n not in succ for n in universe):
            raise Unstructurable('a try region refers to a block that no longer exists')
        aug = {n: list(succ[n]) + ([h] if (n in reach and blocks[n].tag == h) else []) for n in universe}
        j = _post_idom(universe, aug).get(h)
        join = None if j in (None, END) else j
        handler = _reach(h, succ, {join} if join is not None else set())
    if body & handler:
        raise Unstructurable(f'the try body and handler {h} share code')
    outer = blocks[h].tag
    if any(blocks[n].tag >= 0 and blocks[n].tag != outer for n in handler):
        return None
    for n in handler:
        if blocks[n].tag != outer and not _nothrow_block(blocks[n]):
            raise Unstructurable(f'uncovered code that can throw inside handler {h}')
    universe = reach | _reach(h, succ)

    for group, name in ((body, 'try body'), (handler, 'catch')):
        for n in group:
            for s in succ[n]:
                if s not in group and s != join:
                    raise Unstructurable(f'the {name} of handler {h} leaves to {s}, not to the join {join}')
        entry_of = top if group is body else h
        for n in universe - group:
            if any(s in group and s != entry_of for s in succ.get(n, ())):
                raise Unstructurable(f'the {name} of handler {h} is entered other than at its start')
        if group is handler and any(s == h for n in universe - group for s in succ.get(n, ())):
            raise Unstructurable(f'handler {h} is also reachable without an exception')

    fall = min(min(blocks), 0) - 1

    def nested(nodes, start):
        sub = {}
        for n in nodes:
            b = blocks[n]
            sub[n] = SBlock(n, b.stmts, b.term.retarget(lambda t: fall if t == join else t), -1)
        sub[fall] = SBlock(fall, [], STerm('fall'), -1)
        return Graph(start, sub, False)

    body_stmts = structure(nested(body, top))
    catch_stmts = structure(nested(handler, h))
    store = make('ExpressionStatement', expression=make(
        'AssignmentExpression', operator='=', left=_ident('s0'), right=_ident('vm_e')))
    stmt = make('TryStatement', block=make('BlockStatement', body=body_stmts),
                handler=make('CatchClause', param=_ident('vm_e'),
                             body=make('BlockStatement', body=[store] + catch_stmts)),
                finalizer=None)
    out = {b: blk for b, blk in blocks.items() if b not in body and b not in handler}
    out[top] = SBlock(top, [stmt], STerm('goto', target=join) if join is not None else STerm('fall'), outer)
    return out


def prepare(graph, structure, thread):
    """split + thread + contract: a tag-free graph with the try statements built in."""
    g = split_handlers(graph)
    g = Graph(g.entry, thread(g), True)
    return contract_try_regions(g, structure)
