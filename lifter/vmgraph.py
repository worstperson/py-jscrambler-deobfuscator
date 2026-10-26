"""
vmgraph -- the block graph shared by the structurer (vmstructure) and its
exception handling (vmexc).

A Graph is {block id: SBlock}; a block is a run of statements ending in one
terminator (STerm):

    goto target | branch cond t f | return value | throw value | fall

`fall` is the end of a nested region: control leaves the statements built for it
and continues after them. vmexc uses it for the bodies of try/catch.

`SBlock.tag` is the exception handler covering the block's statements (a block
id), or -1. `Graph.has_exc` says some blocks carry one.
"""


class Unstructurable(Exception):
    pass


class STerm:
    __slots__ = ('kind', 'cond', 't', 'f', 'target', 'value')

    def __init__(self, kind, cond=None, t=None, f=None, target=None, value=None):
        self.kind, self.cond, self.t, self.f = kind, cond, t, f
        self.target, self.value = target, value

    def successors(self):
        if self.kind == 'goto':
            return [self.target]
        if self.kind == 'branch':
            return [self.t] if self.t == self.f else [self.t, self.f]
        return []

    def retarget(self, mapping):
        """A copy whose targets are passed through mapping (a dict or callable)."""
        m = mapping if callable(mapping) else (lambda b: mapping.get(b, b))
        if self.kind == 'goto':
            return STerm('goto', target=m(self.target))
        if self.kind == 'branch':
            return STerm('branch', cond=self.cond, t=m(self.t), f=m(self.f))
        return self


class SBlock:
    __slots__ = ('bid', 'stmts', 'term', 'tag')

    def __init__(self, bid, stmts, term, tag=-1):
        self.bid, self.stmts, self.term, self.tag = bid, stmts, term, tag

    def copy(self, **changes):
        out = SBlock(self.bid, self.stmts, self.term, self.tag)
        for k, v in changes.items():
            setattr(out, k, v)
        return out


class Graph:
    def __init__(self, entry, blocks, has_exc=False):
        self.entry = entry
        self.blocks = blocks            # {bid: SBlock}
        self.has_exc = has_exc
