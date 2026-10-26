"""
vmexpr_liveness -- structured liveness of local variables.
"""

from jsir import walk
from lifter.transformers.vmexpr_util import LOOP_KINDS, assignment_of, names_read


class _Liveness:
    def __init__(self, tracked):
        self.tracked = tracked
        self.after = {}
        self.ctx = []
        self.unsafe = False
        self.label = None

    def reads(self, node):
        return names_read(node) & self.tracked.keys()

    def block(self, stmts, live):
        for s in reversed(list(stmts)):
            live = self.stmt(s, live)
        return live

    def sub(self, n, live):
        if n is None:
            return live
        return self.block(n.children_in('body'), live) if n.kind == 'BlockStatement' else self.stmt(n, live)

    def stmt(self, s, out):
        self.after[s.nid] = frozenset(out)
        k = s.kind
        if k == 'ExpressionStatement':
            d = assignment_of(s)
            if d and d[0] in self.tracked:
                return (set(out) - {d[0]}) | self.reads(d[1].get('right'))
            return set(out) | self.reads(s.get('expression'))
        if k == 'VariableDeclaration':
            live = set(out)
            for decl in reversed(s.children_in('declarations')):
                if decl.get('init') is not None:
                    live = (live - {decl.get('id').attr('name')}) | self.reads(decl.get('init'))
            return live
        if k in ('ReturnStatement', 'ThrowStatement'):
            return self.reads(s.get('argument')) if s.get('argument') is not None else set()
        if k == 'EmptyStatement':
            return set(out)
        if k == 'BlockStatement':
            return self.block(s.children_in('body'), out)
        if k == 'IfStatement':
            then = self.sub(s.get('consequent'), set(out))
            other = self.sub(s.get('alternate'), set(out)) if s.get('alternate') is not None else set(out)
            return self.reads(s.get('test')) | then | other
        if k == 'TryStatement':
            handler = s.get('handler')
            catch_live = self.sub(handler.get('body'), set(out)) if handler is not None else set()
            block_out = set(out) | catch_live
            finalizer = s.get('finalizer')
            if finalizer is not None:
                block_out = self.sub(finalizer, block_out)
            return self.sub(s.get('block'), block_out)
        if k == 'LabeledStatement':
            name = s.get('label').attr('name')
            body = s.get('body')
            if body.kind in LOOP_KINDS:
                self.label = name
                return self.stmt(body, out)
            self.ctx.append({'labels': {name}, 'loop': False, 'brk': set(out), 'cont': None})
            live = self.sub(body, out)
            self.ctx.pop()
            return live
        if k in ('WhileStatement', 'DoWhileStatement'):
            return self.loop(s, out)
        if k in ('BreakStatement', 'ContinueStatement'):
            label = s.get('label')
            want = 'brk' if k == 'BreakStatement' else 'cont'
            for c in reversed(self.ctx):
                if (label is None and c['loop']) or (label is not None and label.attr('name') in c['labels']):
                    return set(c[want]) if c[want] is not None else set(self.tracked)
            self.unsafe = True
            return set(self.tracked)
        self.unsafe = True
        return set(self.tracked)

    def loop(self, s, out):
        labels = {self.label} if self.label else set()
        self.label = None
        test = s.get('test')
        endless = test.kind == 'Literal' and test.attr('value') is True
        exits = set() if endless else set(out)
        head = set()
        while True:
            if s.kind == 'WhileStatement':
                ctx = {'labels': labels, 'loop': True, 'brk': set(out), 'cont': head}
                self.ctx.append(ctx)
                body_in = self.sub(s.get('body'), set(head))
                self.ctx.pop()
                new = self.reads(test) | exits | body_in
            else:
                test_in = self.reads(test) | head | exits
                ctx = {'labels': labels, 'loop': True, 'brk': set(out), 'cont': test_in}
                self.ctx.append(ctx)
                body_in = self.sub(s.get('body'), test_in)
                self.ctx.pop()
                new = body_in
            if new == head:
                return head
            head = new


def _escapes(n, in_loop=False, labels=frozenset()):
    k = n.kind
    if k in ('BreakStatement', 'ContinueStatement'):
        label = n.get('label')
        return (not in_loop) if label is None else label.attr('name') not in labels
    if k == 'LabeledStatement':
        labels = labels | {n.get('label').attr('name')}
    if k in LOOP_KINDS:
        in_loop = True
    return any(_escapes(c, in_loop, labels) for _s, c, _r in n.iter_children())
