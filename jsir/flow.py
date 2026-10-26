"""
Control flow.

A CFG per function body, built over statements. Expressions are not split into
basic blocks: `a() ? b() : c()` stays one instruction. That is a deliberate
limit -- splitting expression-level control flow would double the size of this
module and buy precision that matters for compilers optimising real programs,
not for a deobfuscator whose input is already `switch`-flattened statements.
Where expression-level branching matters, the value engine handles it inside a
single instruction.

The dominator computation is Cooper-Harvey-Kennedy, iterative. The existing one
in controlFlowUnflattener is the same algorithm with a recursive DFS, which is a
stack overflow waiting for a large enough flattened function; these samples
routinely have hundreds of switch cases.
"""

from .schema import FUNCTION_KINDS, LOOP_KINDS


ENTRY = 'entry'
EXIT = 'exit'


class Block:
    __slots__ = ('bid', 'statements', 'successors', 'predecessors', 'kind', 'source')

    def __init__(self, bid, kind='plain', source=None):
        self.bid = bid
        self.kind = kind
        self.source = source
        self.statements = []
        self.successors = []      # (label, block_id)
        self.predecessors = []

    def __repr__(self):
        return f'<Block {self.bid} {self.kind} n={len(self.statements)}>'

    def successor(self, label):
        """The target bid of this block's edge carrying `label`, or None.
        For a block with more than one edge sharing a label (not the usual
        case, but nothing stops a caller from calling connect() twice),
        returns the first."""
        for edge_label, bid in self.successors:
            if edge_label == label:
                return bid
        return None


class CFG:
    def __init__(self, owner):
        self.owner = owner              # function node, or Program
        self.blocks = {}
        self.entry = None
        self.exit = None
        self._counter = 0
        self.block_of_statement = {}

    def new_block(self, kind='plain', source=None):
        self._counter += 1
        bid = self._counter
        block = Block(bid, kind, source)
        self.blocks[bid] = block
        return block

    def connect(self, source, target, label='next'):
        if source is None or target is None:
            return
        if any(existing == (label, target.bid) for existing in source.successors):
            return
        source.successors.append((label, target.bid))
        target.predecessors.append((label, source.bid))

    def successors(self, bid):
        return [target for _label, target in self.blocks[bid].successors]

    def predecessors(self, bid):
        return [source for _label, source in self.blocks[bid].predecessors]

    def reachable(self):
        seen = set()
        stack = [self.entry.bid]
        while stack:
            bid = stack.pop()
            if bid in seen:
                continue
            seen.add(bid)
            stack.extend(self.successors(bid))
        return seen

    def reverse_postorder(self):
        order = []
        seen = set()
        stack = [(self.entry.bid, False)]
        while stack:
            bid, processed = stack.pop()
            if processed:
                order.append(bid)
                continue
            if bid in seen:
                continue
            seen.add(bid)
            stack.append((bid, True))
            for target in reversed(self.successors(bid)):
                if target not in seen:
                    stack.append((target, False))
        order.reverse()
        return order


class _Builder:
    def __init__(self, owner):
        self.cfg = CFG(owner)
        self.cfg.entry = self.cfg.new_block('entry')
        self.cfg.exit = self.cfg.new_block('exit')
        self.breaks = []        # (label_name_or_None, block)
        self.continues = []
        self.returns = [self.cfg.exit]
        self.throws = [self.cfg.exit]

    def emit(self, block, statement):
        block.statements.append(statement)
        self.cfg.block_of_statement[statement.nid] = block

    def build(self, body_statements):
        tail = self.run(body_statements, self.cfg.entry)
        if tail is not None:
            self.cfg.connect(tail, self.cfg.exit, 'fallthrough')
        return self.cfg

    def run(self, statements, current):
        for statement in statements:
            current = self.statement(statement, current)
            if current is None:
                break        # rest is unreachable
        return current

    def statement(self, node, current):
        kind = node.kind
        handler = getattr(self, f'_on_{kind}', None)
        if handler is None:
            self.emit(current, node)
            return current
        return handler(node, current)

    # -- straight line ------------------------------------------------------

    def _on_FunctionDeclaration(self, node, current):
        return current          # hoisted; its body is a separate CFG

    def _on_EmptyStatement(self, node, current):
        return current

    def _on_BlockStatement(self, node, current):
        return self.run(node.children_in('body'), current)

    # -- branches -----------------------------------------------------------

    def _on_IfStatement(self, node, current):
        # The instruction is the test, not the whole statement: a block holds
        # things that execute in sequence, and the branches do not.
        self.emit(current, node.get('test'))
        join = self.cfg.new_block('join', node)

        then_block = self.cfg.new_block('then', node.get('consequent'))
        self.cfg.connect(current, then_block, 'true')
        then_tail = self.statement(node.get('consequent'), then_block)
        if then_tail is not None:
            self.cfg.connect(then_tail, join)

        alternate = node.get('alternate')
        if alternate is not None:
            else_block = self.cfg.new_block('else', alternate)
            self.cfg.connect(current, else_block, 'false')
            else_tail = self.statement(alternate, else_block)
            if else_tail is not None:
                self.cfg.connect(else_tail, join)
        else:
            self.cfg.connect(current, join, 'false')
        return join

    def _on_SwitchStatement(self, node, current):
        self.emit(current, node.get('discriminant'))
        join = self.cfg.new_block('join', node)
        self.breaks.append((None, join))

        cases = node.children_in('cases')
        case_blocks = [self.cfg.new_block('case', case) for case in cases]

        # Dispatch: case tests run in source order until one matches, then
        # control jumps to that case's body. `default` is taken only after every
        # test has failed, wherever it sits in the source. Each test must be
        # emitted somewhere, otherwise nothing in it is ever evaluated and
        # `case SOME_CONST:` can never be folded.
        dispatch = current
        default_block = None
        for case, case_block in zip(cases, case_blocks):
            test = case.get('test')
            if test is None:
                default_block = case_block
                continue
            test_block = self.cfg.new_block('case-test', case)
            self.cfg.connect(dispatch, test_block, 'next-test')
            self.emit(test_block, test)
            self.cfg.connect(test_block, case_block, 'case')
            dispatch = test_block
        if default_block is not None:
            self.cfg.connect(dispatch, default_block, 'default')
        else:
            self.cfg.connect(dispatch, join, 'no-match')

        previous_tail = None
        for case, case_block in zip(cases, case_blocks):
            if previous_tail is not None:
                self.cfg.connect(previous_tail, case_block, 'fallthrough')
            previous_tail = self.run(case.children_in('consequent'), case_block)
        if previous_tail is not None:
            self.cfg.connect(previous_tail, join)

        self.breaks.pop()
        return join

    # -- loops --------------------------------------------------------------

    def _loop(self, node, current, header_statements, test, body, update, label,
              test_first=True):
        header = self.cfg.new_block('loop-header', node)
        self.cfg.connect(current, header)
        for statement in header_statements:
            self.emit(header, statement)
        if test is not None:
            self.emit(header, test)

        body_block = self.cfg.new_block('loop-body', body)
        exit_block = self.cfg.new_block('loop-exit', node)
        latch = self.cfg.new_block('loop-latch', node)

        self.cfg.connect(header, body_block, 'true' if test is not None else 'next')
        if test is not None:
            self.cfg.connect(header, exit_block, 'false')

        self.breaks.append((label, exit_block))
        self.continues.append((label, latch))
        body_tail = self.statement(body, body_block) if body is not None else body_block
        self.breaks.pop()
        self.continues.pop()

        if body_tail is not None:
            self.cfg.connect(body_tail, latch)
        if update is not None:
            self.emit(latch, update)
        self.cfg.connect(latch, header, 'back')
        return exit_block

    def _on_WhileStatement(self, node, current):
        return self._loop(node, current, [], node.get('test'), node.get('body'),
                          None, self._pending_label)

    def _on_ForStatement(self, node, current):
        init = node.get('init')
        if init is not None:
            self.emit(current, init)
        return self._loop(node, current, [], node.get('test'), node.get('body'),
                          node.get('update'), self._pending_label)

    def _on_DoWhileStatement(self, node, current):
        # The body always runs once: enter at the body, test at the latch.
        body_block = self.cfg.new_block('loop-body', node.get('body'))
        exit_block = self.cfg.new_block('loop-exit', node)
        latch = self.cfg.new_block('loop-latch', node)
        self.cfg.connect(current, body_block)

        self.breaks.append((self._pending_label, exit_block))
        self.continues.append((self._pending_label, latch))
        body_tail = self.statement(node.get('body'), body_block)
        self.breaks.pop()
        self.continues.pop()

        if body_tail is not None:
            self.cfg.connect(body_tail, latch)
        test = node.get('test')
        if test is not None:
            self.emit(latch, test)
        self.cfg.connect(latch, body_block, 'back')
        self.cfg.connect(latch, exit_block, 'false')
        return exit_block

    def _on_ForInStatement(self, node, current):
        return self._iteration_loop(node, current)

    def _on_ForOfStatement(self, node, current):
        return self._iteration_loop(node, current)

    def _iteration_loop(self, node, current):
        self.emit(current, node.get('right'))
        header = self.cfg.new_block('loop-header', node)
        self.cfg.connect(current, header)
        body_block = self.cfg.new_block('loop-body', node.get('body'))
        exit_block = self.cfg.new_block('loop-exit', node)
        self.cfg.connect(header, body_block, 'iterate')
        self.cfg.connect(header, exit_block, 'done')

        self.emit(body_block, node.get('left'))

        self.breaks.append((self._pending_label, exit_block))
        self.continues.append((self._pending_label, header))
        body_tail = self.statement(node.get('body'), body_block)
        self.breaks.pop()
        self.continues.pop()
        if body_tail is not None:
            self.cfg.connect(body_tail, header, 'back')
        return exit_block

    _pending_label = None

    def _on_LabeledStatement(self, node, current):
        name = node.get('label').attr('name')
        body = node.get('body')
        if body.kind in LOOP_KINDS:
            previous = self._pending_label
            self._pending_label = name
            try:
                return self.statement(body, current)
            finally:
                self._pending_label = previous
        join = self.cfg.new_block('join', node)
        self.breaks.append((name, join))
        tail = self.statement(body, current)
        self.breaks.pop()
        if tail is not None:
            self.cfg.connect(tail, join)
        return join

    # -- transfers ----------------------------------------------------------

    def _on_BreakStatement(self, node, current):
        self.emit(current, node)
        target = self._find_target(self.breaks, node.get('label'))
        self.cfg.connect(current, target or self.cfg.exit, 'break')
        return None

    def _on_ContinueStatement(self, node, current):
        self.emit(current, node)
        target = self._find_target(self.continues, node.get('label'))
        self.cfg.connect(current, target or self.cfg.exit, 'continue')
        return None

    def _find_target(self, stack, label_node):
        if label_node is None:
            return stack[-1][1] if stack else None
        name = label_node.attr('name')
        for label, block in reversed(stack):
            if label == name:
                return block
        return None

    def _on_ReturnStatement(self, node, current):
        self.emit(current, node)
        target = self.returns[-1]
        self.cfg.connect(current, target, 'return')
        return None

    def _on_ThrowStatement(self, node, current):
        self.emit(current, node)
        target = self.throws[-1]
        self.cfg.connect(current, target, 'throw')
        return None

    def _on_TryStatement(self, node, current):
        finalizer = node.get('finalizer')
        final_block = self.cfg.new_block('finally', finalizer) if finalizer is not None else None
        join = self.cfg.new_block('join', node)
        try_block = self.cfg.new_block('try', node.get('block'))
        self.cfg.connect(current, try_block)
        if final_block is not None:
            self.returns.append(final_block)
            self.throws.append(final_block)

        handler = node.get('handler')
        if handler is not None:
            handler_block = self.cfg.new_block('catch', handler)
            self.cfg.connect(current, handler_block, 'throw')
            self.cfg.connect(try_block, handler_block, 'throw')
            self.throws.append(handler_block)
            try:
                try_tail = self.statement(node.get('block'), try_block)
            finally:
                self.throws.pop()
            handler_tail = self.statement(handler.get('body'), handler_block)
            if handler_tail is not None:
                self.cfg.connect(handler_tail, join)
            if final_block is not None:
                self.cfg.connect(handler_block, final_block, 'throw')
        else:
            if final_block is not None:
                self.cfg.connect(current, final_block, 'throw')
                self.cfg.connect(try_block, final_block, 'throw')
            try_tail = self.statement(node.get('block'), try_block)

        if final_block is not None:
            self.throws.pop()
            self.returns.pop()
        if try_tail is not None:
            self.cfg.connect(try_tail, join)

        if final_block is not None:
            if join.predecessors:
                self.cfg.connect(join, final_block)
            final_tail = self.statement(finalizer, final_block)
            if final_tail is not None:
                if any(label == 'return' for label, _ in final_block.predecessors):
                    self.cfg.connect(final_tail, self.returns[-1], 'return')
                if any(label == 'throw' for label, _ in final_block.predecessors):
                    self.cfg.connect(final_tail, self.throws[-1], 'throw')
                if join.predecessors:
                    return final_tail
            return None

        return join if join.predecessors else None


def build_cfg(owner):
    """owner is a Program or a function node.

    Program.body is a statement list; a function's body is a single node, which
    is a BlockStatement unless it is a concise arrow body.
    """
    if owner.kind == 'Program':
        return _Builder(owner).build(owner.children_in('body'))

    body = owner.get('body')
    if body is None:
        return _Builder(owner).build([])
    if body.kind != 'BlockStatement':
        builder = _Builder(owner)
        cfg = builder.cfg
        builder.emit(cfg.entry, body)
        cfg.connect(cfg.entry, cfg.exit, 'return')
        return cfg
    return _Builder(owner).build(body.children_in('body'))


def function_bodies(root):
    """Every unit that gets its own CFG, outermost first."""
    yield root
    stack = [root]
    while stack:
        node = stack.pop()
        for _slot, child, _role in node.iter_children():
            if child.kind in FUNCTION_KINDS:
                yield child
            stack.append(child)


# ---------------------------------------------------------------------------
# Dominators
# ---------------------------------------------------------------------------

class Dominators:
    def __init__(self, idom, order):
        self.idom = idom
        self.order = order

    def dominates(self, a, b):
        if b not in self.idom:
            return False
        current = b
        while True:
            if current == a:
                return True
            parent = self.idom.get(current)
            if parent is None or parent == current:
                return False
            current = parent

    def frontier(self, cfg):
        frontier = {bid: set() for bid in self.idom}
        for bid in self.idom:
            preds = [p for p in cfg.predecessors(bid) if p in self.idom]
            if len(preds) < 2:
                continue
            for pred in preds:
                runner = pred
                while runner != self.idom[bid] and runner in self.idom:
                    frontier[runner].add(bid)
                    nxt = self.idom[runner]
                    if nxt == runner:
                        break
                    runner = nxt
        return frontier

    def natural_loops(self, node_ids, successors, predecessors):
        """Natural loops via back edges, over this dominance relation.

        An edge u -> v is a back edge exactly when v dominates u; v is then
        a loop header and u one of its latches. A header's node set is
        everything that reaches a latch by walking predecessors without
        leaving the set of nodes this dominance relation actually covers
        (nodes outside `self.idom` are unreachable from the entry this
        relation was computed for, so they can't be part of a loop in it).

        A header with more than one back edge into it (multiple latches)
        gets one merged entry, as is standard for natural-loop analysis
        over a reducible graph -- which a structured-then-flattened
        switch/loop always is.

        `node_ids`, `successors` and `predecessors` follow the same
        protocol as `dominators()`: `successors`/`predecessors` are
        callables from a node id to a list of node ids, so this works
        equally over a real CFG's `.successors`/`.predecessors` or over an
        ad hoc graph described by plain dicts/lambdas.

        Returns {header: {'latches': [...], 'nodes': {...}}}.
        """
        latches_by_header = {}
        for u in node_ids:
            for v in successors(u):
                if v in self.idom and self.dominates(v, u):
                    latches_by_header.setdefault(v, []).append(u)
        result = {}
        for header, latches in latches_by_header.items():
            loop_set = {header}
            stack = list(latches)
            while stack:
                n = stack.pop()
                if n not in loop_set and n in self.idom:
                    loop_set.add(n)
                    stack.extend(predecessors(n))
            result[header] = {'latches': latches, 'nodes': loop_set}
        return result


def dominators(cfg, entry=None, successors=None, predecessors=None):
    entry_id = entry if entry is not None else cfg.entry.bid
    successors = successors or cfg.successors
    predecessors = predecessors or cfg.predecessors

    order = _reverse_postorder(entry_id, successors)
    rank = {bid: index for index, bid in enumerate(order)}
    idom = {entry_id: entry_id}

    def intersect(a, b):
        while a != b:
            while rank[a] > rank[b]:
                a = idom[a]
            while rank[b] > rank[a]:
                b = idom[b]
        return a

    changed = True
    while changed:
        changed = False
        for bid in order:
            if bid == entry_id:
                continue
            candidates = [p for p in predecessors(bid) if p in idom and p in rank]
            if not candidates:
                continue
            new_idom = candidates[0]
            for candidate in candidates[1:]:
                new_idom = intersect(new_idom, candidate)
            if idom.get(bid) != new_idom:
                idom[bid] = new_idom
                changed = True
    return Dominators(idom, order)


def _reverse_postorder(entry_id, successors):
    order = []
    seen = set()
    stack = [(entry_id, False)]
    while stack:
        bid, processed = stack.pop()
        if processed:
            order.append(bid)
            continue
        if bid in seen:
            continue
        seen.add(bid)
        stack.append((bid, True))
        for target in reversed(successors(bid)):
            if target not in seen:
                stack.append((target, False))
    order.reverse()
    return order
