"""
vmdisasm -- generic disassembler for this VM family.

Nothing here knows an opcode number or an operand encoding. Both come from the
sample itself:

  * operands are decoded by running the build's OWN reader helpers (the byte /
    varint / 32-bit functions vmsummary classified) in the vmloader machine,
    with the pc and key variables set per instruction;
  * what an instruction does to control flow comes from its vmsummary template:
    `$jump(t, k)`, `$setkey(k)`, `$return` / `$throw`, and whether a jump sits
    under an `if`.

Decoding is path-dependent (the key is part of the state and travels along
jump / setkey edges), so this is a recursive-descent sweep over (pc, key)
states, not a linear scan. Exception-table entries add implicit edges to their
handlers. The result is instructions, edges and basic blocks, ready for a
structurer.

Usage:  python vmdisasm.py SCRIPT.js [--children] [--summaries]
"""

import sys
from collections import defaultdict

from jsir import values as V
from lifter.vmsummary import show, show_stmt, _name
from lifter import vmloader as L

KIND = {'b': 'byte', 'w': 'word', 'v': 'varint'}
CONTROL = ('$jump', '$setkey', '$return', '$throw')


class Edge:
    __slots__ = ('kind', 'pc', 'key', 'label', 'cond')

    def __init__(self, kind, pc, key, label=None, cond=None):
        self.kind, self.pc, self.key, self.label, self.cond = kind, pc, key, label, cond   # cond: True/False arm, None if unconditional

    def __repr__(self):
        return f'{self.kind}->{self.pc:#06x}/{self.key:#04x}'


class Instruction:
    def __init__(self, pc, key):
        self.pc, self.key = pc, key
        self.opcode = None
        self.operands = {}
        self.size = 0
        self.summary = None
        self.edges = []          # normal control flow
        self.exc = []            # implicit edges to exception handlers
        self.child = None        # constant index of a child VM this instruction instantiates
        self.error = None

    @property
    def next_pc(self):
        return self.pc + self.size

    @property
    def terminal(self):
        return not self.edges and self.error is None


class Disassembly:
    def __init__(self, decoded):
        self.decoded = decoded
        self.instructions = {}          # (pc, key) -> Instruction
        self.preds = defaultdict(list)  # state -> [(src pc, src key, edge kind)]
        self.notes = []                 # problems worth a look
        self.info = []                  # expected quirks: return stubs hidden inside other instructions

    # -- queries ---------------------------------------------------------

    def ordered(self):
        return sorted(self.instructions.values(), key=lambda i: (i.pc, i.key))

    def covered(self):
        spans = sorted((i.pc, i.next_pc) for i in self.instructions.values() if i.size)
        merged = []
        for a, b in spans:
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        return merged

    def uncovered(self):
        gaps, cursor, end = [], 0, len(self.decoded.code)
        for a, b in self.covered():
            if a > cursor:
                gaps.append((cursor, a))
            cursor = max(cursor, b)
        if cursor < end:
            gaps.append((cursor, end))
        return gaps

    def blocks(self):
        """Basic blocks as lists of Instructions, keyed by their first (pc, key) state."""
        leaders = {(0, self.decoded.key)}
        for i in self.instructions.values():
            branching = len(i.edges) > 1 or any(e.kind == 'jump' for e in i.edges)
            for e in i.edges:
                if branching:
                    leaders.add((e.pc, e.key))
            for e in i.exc:
                leaders.add((e.pc, e.key))
        leaders = {st for st in leaders if st in self.instructions}
        out = {}
        for start in sorted(leaders):
            block, state = [], start
            while state in self.instructions:
                instr = self.instructions[state]
                block.append(instr)
                if len(instr.edges) != 1 or instr.edges[0].kind != 'fall':
                    break
                state = (instr.edges[0].pc, instr.edges[0].key)
                if state in leaders:
                    break
            if block:
                out[start] = block
        return out


class Disassembler:
    MAX_STATES = 50_000

    def __init__(self, layout, analysis, decoded):
        self.layout, self.analysis, self.decoded = layout, analysis, decoded
        it = analysis.interp
        self.it = it
        self.machine, factory_env = decoded.machine, decoded.env
        self.env = L.Env(factory_env)
        self.machine.hoist(it.func.get('body').children_in('body'), self.env)
        factory_env.vars[it.bytes_name].trace = None          # stop recording reads
        self.readers = {h.kind: self.env.vars[name] for name, h in it.helpers.items()
                        if h.kind in ('byte', 'word', 'varint')}
        self.fetch = self.env.vars[it.fetch_name]
        self.code_len = len(decoded.code)
        self.base = decoded.code_base
        self.notes = []

    # -- one instruction -------------------------------------------------

    def decode_at(self, pc, key):
        instr = Instruction(pc, key)
        if not 0 <= pc < self.code_len:
            instr.error = 'pc outside the code section'
            return instr
        it, env = self.it, self.env
        env.vars[it.pc_name] = float(pc + self.base)
        env.vars[it.key_name] = float(key)
        self.machine.budget = 200_000
        try:
            instr.opcode = int(self.machine.call(self.fetch, []))
            summary = self.analysis.summaries.get(instr.opcode)
            if summary is None:
                instr.error = f'unknown opcode {instr.opcode}'
                return instr
            instr.summary = summary
            for name in summary.layout:
                instr.operands[name] = int(self.machine.call(self.readers[KIND[name[0]]], []))
        except (L.LoaderError, KeyError, ValueError, OverflowError, TypeError) as exc:
            instr.error = f'decode failed: {exc}'
            return instr
        instr.size = int(env.vars[it.pc_name]) - self.base - pc
        self._control(instr)
        return instr

    # -- control flow from the summary ------------------------------------

    def _eval(self, n, ops):
        v = self._eval_raw(n, ops)
        return v if isinstance(v, float) and v == v and abs(v) != float('inf') else None

    def _eval_raw(self, n, ops):
        k = n.kind
        if k == 'Literal':
            return n.attr('value')
        if k == 'Identifier' and n.attr('name')[1:] in ops and n.attr('name').startswith('$'):
            return float(ops[n.attr('name')[1:]])
        if k == 'UnaryExpression':
            v = self._eval(n.get('argument'), ops)
            return None if v is None else V.unary(n.attr('operator'), v)
        if k == 'BinaryExpression':
            a, b = self._eval(n.get('left'), ops), self._eval(n.get('right'), ops)
            return None if a is None or b is None else V.binary(n.attr('operator'), a, b)
        return None

    @staticmethod
    def _effects(statements):
        eff = {'jump': None, 'setkey': None, 'terminal': False}
        for s in statements:
            if s.kind != 'ExpressionStatement' or s.get('expression').kind != 'CallExpression':
                continue
            call = s.get('expression')
            name, args = _name(call.get('callee')), call.children_in('arguments')
            if name == '$jump':
                eff['jump'] = (args[0], args[1])
            elif name == '$setkey':
                eff['setkey'] = args[0]
            elif name in ('$return', '$throw'):
                eff['terminal'] = True
        return eff

    def _control(self, instr):
        summary, ops = instr.summary, instr.operands
        body = summary.body
        base = self._effects(body)
        ifs = [s for s in body if s.kind == 'IfStatement']
        temps = {s.children_in('declarations')[0].get('id').attr('name'):
                 s.children_in('declarations')[0].get('init')
                 for s in body if s.kind == 'VariableDeclaration'}
        hook = self.hook(instr, temps)
        paths = []
        if len(ifs) > 1:
            self.note(instr, 'more than one conditional in a handler; control flow not modelled')
        if ifs:
            test = ifs[0].get('test')
            label = show(test, hook=hook)
            then_eff = self._effects(ifs[0].get('consequent').children_in('body'))
            alt = ifs[0].get('alternate')
            else_eff = self._effects(alt.children_in('body')) if alt is not None else {'jump': None, 'setkey': None, 'terminal': False}
            paths = [(label, then_eff), (f'!{label}' if not label.startswith('!') else label[1:], else_eff)]
        else:
            paths = [(None, {'jump': None, 'setkey': None, 'terminal': False})]
        for arm, (label, eff) in enumerate(paths):
            cond = None if len(paths) == 1 else (arm == 0)
            jump = eff['jump'] or base['jump']
            setkey = eff['setkey'] or base['setkey']
            if eff['terminal'] or base['terminal']:
                continue
            if jump is not None:
                target, key = self._eval(jump[0], ops), self._eval(jump[1], ops)
                if target is None or key is None:
                    self.note(instr, 'jump target/key not resolvable from operands')
                    continue
                instr.edges.append(Edge('jump', int(target), int(key), label, cond))
            else:
                key = instr.key
                if setkey is not None:
                    value = self._eval(setkey, ops)
                    if value is None:
                        self.note(instr, 'key not resolvable from operands')
                        continue
                    key = int(value)
                instr.edges.append(Edge('fall', instr.next_pc, key, label, cond))
        # implicit exception edges: the lookup scans the table from the last entry backwards
        if self._may_throw(summary):
            for start, end, handler, hkey in reversed(self.decoded.exceptions):
                if start < instr.next_pc <= end:
                    instr.exc.append(Edge('exc', handler, hkey))
                    break
        for node in self._walk_calls(body):
            if _name(node.get('callee')) == '$child_vm':
                idx = self._eval(node.children_in('arguments')[0].children_in('arguments')[0], ops) \
                    if node.children_in('arguments')[0].kind == 'CallExpression' else None
                instr.child = int(idx) if idx is not None else None

    @staticmethod
    def _walk_calls(body):
        from jsir import walk
        for s in body:
            for n in walk(s, kinds={'CallExpression'}):
                yield n

    @staticmethod
    def _may_throw(summary):
        for s in summary.body:
            if s.kind == 'VariableDeclaration':
                return True                      # every impure operation is bound to a temporary
            if s.kind == 'ExpressionStatement':
                e = s.get('expression')
                if e.kind == 'AssignmentExpression' or _name(e.get('callee') if e.kind == 'CallExpression' else None) == '$throw':
                    return True
        return False

    def note(self, instr, text):
        self.notes.append(f'{instr.pc:#06x}: {text}')

    # -- rendering hook ----------------------------------------------------

    def const_text(self, idx):
        consts = self.decoded.constants
        if idx is None or not 0 <= int(idx) < len(consts):
            return f'<const {idx}?>'
        c = consts[int(idx)]
        if isinstance(c, str):
            import json
            return json.dumps(c) if len(c) <= 40 else f'<str {len(c)} chars>'
        if isinstance(c, float):
            return V.number_to_string(c)
        if c is V.NULL:
            return 'null'
        if c is V.UNDEF:
            return 'undefined'
        return str(c).lower() if isinstance(c, bool) else repr(c)

    def hook(self, instr, temps=None):
        ops = instr.operands

        def hook(n):
            k = n.kind
            if k == 'Identifier':
                name = n.attr('name')
                if name.startswith('$') and name[1:] in ops:
                    return str(ops[name[1:]])
                if temps and name in temps:
                    return '(' + show(temps[name], hook=hook) + ')'
            elif k == 'CallExpression':
                cname, args = _name(n.get('callee')), n.children_in('arguments')
                if cname == '$const' and len(args) == 1:
                    return self.const_text(self._eval(args[0], ops))
                if cname == '$child_vm' and args:
                    first = args[0]
                    idx = self._eval(first.children_in('arguments')[0], ops) \
                        if first.kind == 'CallExpression' and _name(first.get('callee')) == '$const' else None
                    rest = ', '.join(show(a, hook=hook) for a in args[1:])
                    return f'$child_vm(<fn const[{int(idx) if idx is not None else "?"}]>{", " + rest if rest else ""})'
                if cname == '$jump' and len(args) == 2:
                    t, key = self._eval(args[0], ops), self._eval(args[1], ops)
                    if t is not None and key is not None:
                        return f'$jump({int(t):#06x}, key={int(key):#04x})'
                if cname == '$setkey' and len(args) == 1:
                    key = self._eval(args[0], ops)
                    if key is not None:
                        return f'$setkey({int(key):#04x})'
            return None
        return hook

    def gist(self, instr, width=96):
        temps = {s.children_in('declarations')[0].get('id').attr('name'):
                 s.children_in('declarations')[0].get('init')
                 for s in instr.summary.body if s.kind == 'VariableDeclaration'}
        hook = self.hook(instr)
        parts = []
        for s in instr.summary.body:
            parts += [line.strip() for line in show_stmt(s, '', frozenset(), hook)]
        for p in instr.summary.pushes:
            parts.append(f'push {show(p, frozenset(), hook)};')
        text = ' '.join(parts) or '(no effect)'
        return text if len(text) <= width else text[:width - 3] + '...'

    # -- sweep ---------------------------------------------------------------

    def run(self):
        dis = Disassembly(self.decoded)
        self.notes = dis.notes
        work = [(0, self.decoded.key)]
        while work:
            state = work.pop()
            if state in dis.instructions:
                continue
            if len(dis.instructions) >= self.MAX_STATES:
                dis.notes.append(f'state limit ({self.MAX_STATES}) reached; sweep stopped')
                break
            instr = self.decode_at(*state)
            dis.instructions[state] = instr
            for e in instr.edges + instr.exc:
                if not 0 <= e.pc < self.code_len:
                    dis.notes.append(f'{instr.pc:#06x}: {e.kind} edge to {e.pc:#x} leaves the code section')
                else:
                    dis.preds[(e.pc, e.key)].append((instr.pc, instr.key, e.kind))
                    work.append((e.pc, e.key))
        self._report_overlaps(dis)
        for i in dis.ordered():
            if i.error:
                dis.notes.append(f'{i.pc:#06x} key {i.key:#04x}: {i.error}')
        return dis

    def _report_overlaps(self, dis):
        """Same bytes decoded more than one way: different keys at one pc, or a jump into the middle of
        another instruction. The key is part of the state, so both are legitimate here. If the alternative
        decode is a terminal instruction (return / throw) it is a 'return stub' hidden in another
        instruction's bytes (info); two live decodes of the same bytes is something to look at (warning)."""
        states = sorted((i for i in dis.instructions.values() if i.size), key=lambda i: (i.pc, i.key))
        by_pc = defaultdict(list)
        for i in states:
            by_pc[i.pc].append(i)
        for pc, group in sorted(by_pc.items()):
            if len(group) > 1:
                parts = []
                for i in group:
                    via = sorted({f'{p[0]:#06x}/{p[2]}' for p in dis.preds.get((i.pc, i.key), [])})
                    parts.append(f'key {i.key:#04x} = op{i.opcode}{" (terminal)" if i.terminal else ""} '
                                 f'size {i.size} via {", ".join(via) or "entry"}')
                live = sum(1 for i in group if not i.terminal)
                (dis.notes if live > 1 else dis.info).append(f'{pc:#06x} decodes {len(group)} ways: ' + '; '.join(parts))
        active, reported = [], set()
        for i in states:
            active = [(end, j) for end, j in active if end > i.pc]
            for end, j in active:
                if j.pc != i.pc and (j.pc, i.pc) not in reported:
                    reported.add((j.pc, i.pc))
                    text = (f'{i.pc:#06x} (key {i.key:#04x}, op{i.opcode}{" terminal" if i.terminal else ""}) starts '
                            f'inside the instruction at {j.pc:#06x} (key {j.key:#04x}, op{j.opcode}, {j.size} bytes)')
                    (dis.info if i.terminal or j.terminal else dis.notes).append(text)
            active.append((i.next_pc, i))


def disassemble(layout, analysis, decoded):
    d = Disassembler(layout, analysis, decoded)
    return d.run(), d


def listing(layout, analysis, decoded, summaries=False):
    dis, renderer = disassemble(layout, analysis, decoded)
    total = len(decoded.code)
    covered = sum(b - a for a, b in dis.covered())
    print(f'function {decoded.name!r}  arity={decoded.arity}  key={decoded.key:#04x}  '
          f'{len(dis.instructions)} instructions, {covered}/{total} bytes covered, '
          f'{len(dis.blocks())} basic blocks, {len(dis.info)} overlapping decodes (return stubs), {len(dis.notes)} warnings')
    block_starts = set(dis.blocks())
    for i in dis.ordered():
        mark = '>' if (i.pc, i.key) in block_starts else ' '
        if i.error:
            print(f' {mark}{i.pc:#06x} k={i.key:#04x} !! {i.error}')
            continue
        operands = ' '.join(f'{k}={v:#x}' if v > 255 or v < 0 else f'{k}={v}' for k, v in i.operands.items())
        flow = []
        for e in i.edges:
            if e.kind == 'jump' or (e.label is not None):
                flow.append(f'{e.kind} {e.pc:#06x}' + (f' [{e.label}]' if e.label else ''))
        exc = f'   exc->{[f"{e.pc:#06x}" for e in i.exc]}' if i.exc else ''
        print(f' {mark}{i.pc:#06x} k={i.key:#04x} op{i.opcode:<3} {operands:<22} {renderer.gist(i):<70} {"; ".join(flow)}{exc}')
        if summaries:
            print(i.summary.render())
    for a, b in dis.uncovered():
        before = [i for i in dis.ordered() if i.next_pc <= a]
        guess_key = before[-1].key if before else decoded.key
        spec = renderer.decode_at(a, guess_key)
        if spec.error or not spec.size or spec.next_pc != b:
            print(f'  uncovered bytes {a:#06x}..{b:#06x}: {decoded.code[a:b].hex(" ")}  (no clean decode under key {guess_key:#04x})')
        else:
            print(f'  unreachable bytes {a:#06x}..{b:#06x}: dead tail. Under key {guess_key:#04x} (the preceding code\'s) '
                  f'it is op{spec.opcode}: {renderer.gist(spec)}'
                  + (''.join(f'   -> {e.kind} {e.pc:#06x} key {e.key:#04x}' for e in spec.edges)))
    for note in dis.info:
        print(f'  ~ {note}')
    for note in dis.notes:
        print(f'  ! {note}')
    return dis


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    from lifter.vmsummary import analyse
    session = L.load(argv[1])
    layout, analysis = L.build_layout(session)
    payloads = L.find_payloads(session, analysis.interp)
    if not payloads:
        print('no call of the factory with a literal payload found')
        return 2
    seen = set()

    def walk_tree(decoded, depth, label):
        print(f'\n{"  " * depth}== {label}')
        try:
            listing(layout, analysis, decoded, summaries='--summaries' in argv)
        except Exception as exc:                      # keep going: report and move on
            print(f'  !! disassembly of this function failed: {type(exc).__name__}: {exc}')
        if '--children' in argv and depth < 6:
            for idx, child in L.decode_children(layout, decoded, seen).items():
                walk_tree(child, depth + 1, f'child via constant [{idx}] of {decoded.name!r}')

    for _, payload in payloads:
        walk_tree(L.decode_payload(layout, payload), 0, 'main')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
