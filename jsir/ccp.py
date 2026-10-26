"""
Conditional constant propagation.

Why this shape and not SCCP proper: SCCP is sparse because it walks SSA def-use
edges. Getting SSA means building it and then destroying it again, since the
output of this project is JavaScript source, not machine code, and out-of-SSA is
where the subtle bugs live. This is the dense form -- a per-block environment
joined at merges, iterated to fixpoint -- which costs more passes over each
block and is strictly less precise only in cases involving the same variable
holding different constants on different paths through a loop. For obfuscated
code, that is not where the wins are.

What it keeps from SCCP, and the reason it beats an interpreter that forks at
every branch: edges start *not* executable. A block is only evaluated once
something proves control can reach it, and a variable's value joins only over
edges proved reachable. `if (false) { x = 1 }` never contributes `1` to x. The
previous approach explored both sides of every undecidable branch and joined,
which is sound but loses everything the moment a branch is decidable and the
other side assigns.

What it cannot do, and this is worth being clear about: it will not run a loop
until a hash matches. A fixpoint over a lattice terminates by giving up on
values that change; concrete execution terminates by computing. They answer
different questions, so `Oracle.resolve_call` is where a bounded concrete
executor plugs in for the specific calls worth executing.
"""

from . import values as V
from .binding import Binding
from .flow import build_cfg
from .schema import FUNCTION_KINDS


class Oracle:
    """Everything the solver cannot know from the code alone.

    Subclass or instantiate with tables. Kept as one object so a pass can say
    what it is assuming in one place, instead of the assumption being spread
    across a dozen `if name == '__inspector'` checks.
    """

    def __init__(self, globals_=None, members=None, calls=None, global_aliases=()):
        self.globals = dict(globals_ or {})
        self.members = dict(members or {})        # 'globalThis.a.b' -> value
        self.calls = dict(calls or {})            # 'globalThis.a.b' -> python fn
        self.global_aliases = set(global_aliases) | {'globalThis', 'window', 'self', 'global'}

    def resolve_free_name(self, name):
        if name in self.globals:
            return self.globals[name]
        if name in self.global_aliases:
            return V.Opaque('globalThis')
        return V.UNKNOWN

    def resolve_member(self, base, key):
        if isinstance(base, V.Opaque) and isinstance(key, str):
            return self.members.get(f'{base.label}.{key}', V.UNKNOWN)
        return V.UNKNOWN

    def resolve_call(self, path, arguments):
        handler = self.calls.get(path)
        if handler is None:
            return V.UNKNOWN
        try:
            return handler(*arguments)
        except Exception:
            return V.UNKNOWN


class Facts:
    """Results of one solve, keyed by node identity."""

    def __init__(self):
        self.values = {}            # nid -> value
        self.executable = set()     # (source_bid, label, target_bid)
        self.reachable = set()      # bids
        self.impure = set()         # nids whose evaluation had unmodelled effects

    def value_of(self, node):
        if node.nid in self.impure:
            return V.UNKNOWN
        return self.values.get(node.nid, V.UNSET)

    def constant_of(self, node):
        """The value this node always has, or None if it is not a constant.

        UNSET means the node was never reached, which is not the same as "not
        constant" -- callers that want to delete unreached code should ask
        `reachable` instead of inferring it from here.
        """
        value = self.value_of(node)
        if value in V.LATTICE_MARKERS or not V.is_concrete(value):
            return None
        return value

    def truth_of(self, node):
        return V.truthiness(self.value_of(node))


class Solver:
    def __init__(self, tree, index, oracle=None, pinned=None):
        self.tree = tree
        self.index = index
        self.oracle = oracle or Oracle()
        # Bindings that escape -- so the environment cannot track them -- but
        # whose every write turned out to sit in unreachable code. See
        # `pin_dead_write_bindings`.
        self.pinned = dict(pinned or {})

    def solve(self, owner):
        cfg = build_cfg(owner)
        facts = Facts()
        block_in = {}
        executable_edges = set()
        worklist = [cfg.entry.bid]
        facts.reachable.add(cfg.entry.bid)
        block_in[cfg.entry.bid] = self._initial_environment(owner)
        if owner.kind in FUNCTION_KINDS:
            # Parameter defaults (`function f(a = g(1))`) are not statements in
            # any block, so nothing else would ever visit them.
            for param in owner.children_in('params'):
                self._assign_pattern(param, block_in[cfg.entry.bid], facts)

        visits = 0
        limit = 200 * max(1, len(cfg.blocks))
        while worklist:
            visits += 1
            if visits > limit:
                # Guard against a transfer function that is not monotone. If
                # this fires it is a bug here, not in the input.
                break
            bid = worklist.pop()
            block = cfg.blocks[bid]
            environment = dict(block_in.get(bid, {}))
            outgoing = self._transfer(block, environment, facts, cfg)

            for label, target in block.successors:
                if (bid, label, target) not in outgoing:
                    continue
                executable_edges.add((bid, label, target))
                facts.executable.add((bid, label, target))
                merged, changed = self._merge_into(block_in, target, outgoing[(bid, label, target)])
                if changed or target not in facts.reachable:
                    facts.reachable.add(target)
                    if target not in worklist:
                        worklist.append(target)
        return facts, cfg

    # -- environment --------------------------------------------------------

    def _initial_environment(self, owner):
        environment = {}
        scope = self.index.scope_of(owner)
        for binding in scope.bindings.values():
            if binding.promotable and binding.kind in ('var',):
                environment[binding] = V.UNDEF
        return environment

    @staticmethod
    def _merge_into(block_in, target, incoming):
        existing = block_in.get(target)
        if existing is None:
            block_in[target] = dict(incoming)
            return block_in[target], True
        changed = False
        for key in set(existing) | set(incoming):
            left = existing.get(key, V.UNSET)
            right = incoming.get(key, V.UNSET)
            merged = V.join(left, right)
            if merged is not left:
                existing[key] = merged
                changed = True
        return existing, changed

    # -- transfer -----------------------------------------------------------

    def _transfer(self, block, environment, facts, cfg):
        last_value = V.UNSET
        for instruction in block.statements:
            last_value = self._run(instruction, environment, facts)

        outgoing = {}
        labels = [label for label, _target in block.successors]
        for label, target in block.successors:
            if label in ('true', 'false') and ('true' in labels and 'false' in labels):
                truth = V.truthiness(last_value)
                if truth is not V.UNKNOWN:
                    if (truth and label == 'false') or (not truth and label == 'true'):
                        continue
            outgoing[(block.bid, label, target)] = dict(environment)
        return outgoing

    def _run(self, node, environment, facts):
        parent = node.parent
        if parent is not None and parent.kind in ('ForInStatement', 'ForOfStatement') \
                and parent.get('left') is node:
            # The loop target: a fresh, unknowable value every iteration.
            target = node
            if node.kind == 'VariableDeclaration':
                declarators = node.children_in('declarations')
                target = declarators[0].get('id') if declarators else None
            self._assign(target, V.UNKNOWN, environment, facts)
            return V.UNSET
        kind = node.kind
        if kind == 'VariableDeclaration':
            for declarator in node.children_in('declarations'):
                init = declarator.get('init')
                value = self._eval(init, environment, facts) if init is not None else V.UNDEF
                self._assign(declarator.get('id'), value, environment, facts)
            return V.UNSET
        if kind == 'ExpressionStatement':
            return self._eval(node.get('expression'), environment, facts)
        if kind in ('ReturnStatement', 'ThrowStatement'):
            argument = node.get('argument')
            return self._eval(argument, environment, facts) if argument is not None else V.UNDEF
        if kind in ('BreakStatement', 'ContinueStatement', 'EmptyStatement',
                    'DebuggerStatement', 'FunctionDeclaration'):
            return V.UNSET
        return self._eval(node, environment, facts)

    # -- expressions --------------------------------------------------------

    def _record(self, node, value, facts):
        previous = facts.values.get(node.nid, V.UNSET)
        facts.values[node.nid] = V.join(previous, value)
        return value

    def _eval(self, node, environment, facts):
        if node is None:
            return V.UNDEF
        value = self._eval_raw(node, environment, facts)
        self._record(node, value, facts)
        return value

    def _eval_raw(self, node, environment, facts):
        kind = node.kind

        if kind == 'Literal':
            return node.attr('value')

        if kind == 'Identifier':
            name = node.attr('name')
            if name == 'undefined':
                return V.UNDEF
            if name == 'NaN':
                return float('nan')
            if name == 'Infinity':
                return float('inf')
            binding = self.index.binding_of(node)
            if binding is None:
                return self.oracle.resolve_free_name(name)
            if not binding.promotable:
                return self.pinned.get(binding, V.UNKNOWN)
            return environment.get(binding, V.UNKNOWN)

        if kind == 'UnaryExpression':
            operator = node.attr('operator')
            if operator == 'delete':
                argument = node.get('argument')
                if argument is not None and argument.kind == 'MemberExpression':
                    # Visit the parts that are evaluated, but not the member itself:
                    # `delete a[k]` does not read a[k].
                    self._eval(argument.get('object'), environment, facts)
                    self._member_key(argument, environment, facts)  # evals computed property
                self._taint(node, environment, facts)
                return V.UNKNOWN
            return V.unary(operator, self._eval(node.get('argument'), environment, facts))

        if kind == 'BinaryExpression':
            left = self._eval(node.get('left'), environment, facts)
            right = self._eval(node.get('right'), environment, facts)
            return V.binary(node.attr('operator'), left, right)

        if kind == 'LogicalExpression':
            operator = node.attr('operator')
            left = self._eval(node.get('left'), environment, facts)
            truth = V.truthiness(left)
            if truth is V.UNKNOWN:
                # The right side may or may not run. Evaluating it for its value
                # is fine; committing its side effects is not, so anything it
                # writes is widened.
                self._speculate(node.get('right'), environment, facts)
                return V.UNKNOWN
            short_circuits = (operator == '&&' and not truth) or \
                             (operator == '||' and truth) or \
                             (operator == '??' and left is not V.NULL and left is not V.UNDEF)
            if short_circuits:
                return left
            return self._eval(node.get('right'), environment, facts)

        if kind == 'ConditionalExpression':
            truth = V.truthiness(self._eval(node.get('test'), environment, facts))
            if truth is V.UNKNOWN:
                self._speculate(node.get('consequent'), environment, facts)
                self._speculate(node.get('alternate'), environment, facts)
                return V.UNKNOWN
            taken = 'consequent' if truth else 'alternate'
            skipped = 'alternate' if truth else 'consequent'
            self._record_unreached(node.get(skipped), facts)
            return self._eval(node.get(taken), environment, facts)

        if kind == 'SequenceExpression':
            value = V.UNDEF
            for expression in node.children_in('expressions'):
                value = self._eval(expression, environment, facts)
            return value

        if kind == 'AssignmentExpression':
            right = self._eval(node.get('right'), environment, facts)
            operator = node.attr('operator', '=')
            target = node.get('left')
            if operator != '=':
                current = self._eval(target, environment, facts)
                right = V.binary(operator[:-1], current, right)
            self._assign(target, right, environment, facts)
            return right

        if kind == 'UpdateExpression':
            target = node.get('argument')
            current = self._eval(target, environment, facts)
            number = V.to_number(current)
            if number is V.UNKNOWN:
                self._assign(target, V.UNKNOWN, environment, facts)
                return V.UNKNOWN
            delta = 1.0 if node.attr('operator') == '++' else -1.0
            updated = number + delta
            self._assign(target, updated, environment, facts)
            return updated if node.attr('prefix') else number

        if kind == 'MemberExpression':
            base = self._eval(node.get('object'), environment, facts)
            key = self._member_key(node, environment, facts)
            if key is V.UNKNOWN or base is V.UNKNOWN:
                return V.UNKNOWN
            if isinstance(base, str) and key == 'length':
                return float(len(base))
            return self.oracle.resolve_member(base, key)

        if kind == 'CallExpression':
            arguments = [self._eval(argument, environment, facts)
                         for argument in node.children_in('arguments')]
            #path = self._callee_path(node.get('callee'), environment, facts)
            callee = node.get('callee')
            # _callee_path only walks the callee to build a dotted name for
            # the oracle, and gives up the moment one segment is unresolvable
            # (e.g. a computed key it can't evaluate) -- at which point it
            # returns without ever reaching the rest of the callee, such as
            # the object side of `(a ? b : c)[dynamicKey]`. That object is a
            # real expression with its own foldable subexpressions, and it
            # deserves to be visited and recorded regardless of whether the
            # call itself can be resolved. So evaluate the callee in full
            # first; _callee_path below then re-walks it (redundantly, but
            # cheaply) purely to see if a dotted path happens to fall out.
            self._eval(callee, environment, facts)
            path = self._callee_path(callee, environment, facts)
            # end patch
            result = V.UNKNOWN
            if path is not None:
                result = self.oracle.resolve_call(path, arguments)
            if result is V.UNKNOWN:
                self._taint(node, environment, facts)
            return result

        if kind in FUNCTION_KINDS:
            return V.Opaque(f'fn:{node.nid}')

        if kind == 'TemplateLiteral':
            quasis = node.children_in('quasis')
            expressions = node.children_in('expressions')
            parts = []
            known = True
            for i, quasi in enumerate(quasis):
                parts.append(quasi.attr('cooked', ''))
                if i < len(expressions):
                    text = V.to_string(self._eval(expressions[i], environment, facts))
                    if text is V.UNKNOWN:
                        known = False
                    else:
                        parts.append(text)
            return ''.join(parts) if known else V.UNKNOWN

        if kind in ('ArrayExpression', 'ObjectExpression'):
            for _slot, child, _role in node.iter_children():
                self._eval(child, environment, facts)
                if child is not None and child.kind == 'SpreadElement':
                    # Spreading runs iterators/getters we don't model.
                    self._taint(node, environment, facts)
            # Contents are unmodelled, but it is always a truthy object.
            # Fresh label per evaluation: two evaluations are two objects.
            return V.Opaque(f'obj:{node.nid}')

        if kind in ('NewExpression', 'TaggedTemplateExpression'):
            for _slot, child, _role in node.iter_children():
                self._eval(child, environment, facts)
            return V.UNKNOWN

        if kind == 'ThisExpression':
            return V.UNKNOWN

        for _slot, child, _role in node.iter_children():
            self._eval(child, environment, facts)
        self._taint(node, environment, facts)
        return V.UNKNOWN

    def _member_key(self, node, environment, facts):
        property_node = node.get('property')
        if node.attr('computed', False):
            key = self._eval(property_node, environment, facts)
            return V.to_string(key)
        if property_node is not None and property_node.kind == 'Identifier':
            return property_node.attr('name')
        return V.UNKNOWN

    def _callee_path(self, callee, environment, facts):
        parts = []
        current = callee
        while current is not None and current.kind == 'MemberExpression':
            key = self._member_key(current, environment, facts)
            if key is V.UNKNOWN:
                return None
            parts.append(str(key))
            current = current.get('object')
        if current is None:
            return None
        if current.kind == 'Identifier':
            binding = self.index.binding_of(current)
            name = current.attr('name')
            if binding is None and name in self.oracle.global_aliases:
                parts.append('globalThis')
            elif binding is None or name == self.oracle.runtime_object:
                parts.append(name)
            else:
                value = environment.get(binding, V.UNKNOWN)
                if isinstance(value, V.Opaque):
                    parts.append(value.label)
                else:
                    return None
        else:
            value = self._eval(current, environment, facts)
            if not isinstance(value, V.Opaque):
                return None
            parts.append(value.label)
        return '.'.join(reversed(parts))

    # -- writes -------------------------------------------------------------

    def _assign(self, target, value, environment, facts):
        if target is None:
            return
        if target.kind == 'Identifier':
            binding = self.index.binding_of(target)
            if binding is None:
                return
            if binding.promotable:
                environment[binding] = value
            return
        if target.kind == 'MemberExpression':
            self._eval(target.get('object'), environment, facts)
            self._member_key(target, environment, facts)
            return
        # Destructuring. The source value is not modelled, so every target is
        # unknown from now on -- but the pattern also contains *expressions*
        # (defaults, computed keys, member targets) that must be evaluated and
        # recorded like any other, or they never fold.
        self._assign_pattern(target, environment, facts)

    def _assign_pattern(self, pattern, environment, facts):
        """Walk a binding/assignment pattern in evaluation order.

        Targets become UNKNOWN. Default initialisers (`[a = f(1)] = src`) only
        run when the element is undefined, so they are evaluated speculatively:
        their node values are recorded, their writes are widened. Visiting in
        order means a later default that reads an earlier target sees UNKNOWN.
        """
        if pattern is None:
            return
        kind = pattern.kind
        if kind in ('Identifier', 'MemberExpression'):
            self._assign(pattern, V.UNKNOWN, environment, facts)
        elif kind == 'AssignmentPattern':
            self._speculate(pattern.get('right'), environment, facts)
            self._assign_pattern(pattern.get('left'), environment, facts)
        elif kind == 'RestElement':
            self._assign_pattern(pattern.get('argument'), environment, facts)
        elif kind == 'ArrayPattern':
            for element in pattern.children_in('elements'):
                self._assign_pattern(element, environment, facts)
        elif kind == 'ObjectPattern':
            for prop in pattern.children_in('properties'):
                if prop.kind == 'Property':
                    if prop.attr('computed', False):
                        self._eval(prop.get('key'), environment, facts)
                    self._assign_pattern(prop.get('value'), environment, facts)
                else:  # RestElement
                    self._assign_pattern(prop, environment, facts)
        else:
            # Unrecognised shape: stay conservative.
            for node in _identifiers_in(pattern):
                binding = self.index.binding_of(node)
                if binding is not None and binding.promotable:
                    environment[binding] = V.UNKNOWN

    def _taint(self, node, environment, facts):
        """An unmodelled effect. Anything a closure could reach is now unknown."""
        facts.impure.add(node.nid)
        for binding in list(environment):
            if binding.escapes:
                environment[binding] = V.UNKNOWN

    def _speculate(self, node, environment, facts):
        """Evaluate for value, discard the writes, widen what it wrote."""
        if node is None:
            return
        snapshot = dict(environment)
        self._eval(node, environment, facts)
        for binding in set(environment) | set(snapshot):
            before = snapshot.get(binding, V.UNSET)
            after = environment.get(binding, V.UNSET)
            environment[binding] = before if before is after else V.UNKNOWN

    @staticmethod
    def _record_unreached(node, facts):
        return


def _identifiers_in(node):
    stack = [node]
    while stack:
        current = stack.pop()
        if current.kind == 'Identifier':
            yield current
        for _slot, child, _role in current.iter_children():
            stack.append(child)


def solve_all(tree, index, oracle=None, pinned=None):
    """Solve every function body. Returns {owner_nid: (Facts, CFG)}."""
    from .flow import function_bodies
    solver = Solver(tree, index, oracle, pinned)
    results = {}
    for owner in function_bodies(tree.root):
        if owner.kind in FUNCTION_KINDS or owner.kind == 'Program':
            results[owner.nid] = solver.solve(owner)
    return results


def _owning_block(results, node):
    """The block that executes this node, if any body's CFG covers it."""
    owner = node
    while owner is not None:
        entry = results.get(owner.nid)
        if entry is not None:
            facts, cfg = entry
            probe = node
            while probe is not None and probe is not owner:
                block = cfg.block_of_statement.get(probe.nid)
                if block is not None:
                    return facts, block
                probe = probe.parent
            return facts, None
        owner = owner.parent
    return None, None


def pin_dead_write_bindings(index, results):
    """Escaping bindings whose every write is in unreachable code.

    An escaping variable is normally opaque: a closure could write it at any
    time, so the environment cannot hold a value for it. But if every one of
    those writes sits in a block control never reaches, the variable keeps its
    declared value forever, and that is worth knowing -- it is exactly the shape
    an anti-debug probe leaves behind once the probe itself folds to false.

    This is the honest version of the old resolver's guess. That one asked "is
    this assignment inside a conditional", which is a syntactic proxy; this asks
    whether the assignment is reachable, which is the actual question.
    """
    pinned = {}
    for binding in index.all_bindings():
        if binding.promotable or binding.escapes is False:
            continue
        if not binding.writes or binding.kind not in ('var', 'let', 'param'):
            continue
        all_dead = True
        for write in binding.writes:
            facts, block = _owning_block(results, write)
            if facts is None or block is None or block.bid in facts.reachable:
                all_dead = False
                break
        if all_dead:
            declaration = binding.declaration
            init = declaration.get('init') if declaration is not None and \
                declaration.kind == 'VariableDeclarator' else None
            pinned[binding] = V.UNDEF if init is None else V.UNKNOWN
    return pinned


#def _entry_statement_index(node, entry_statements):
def _entry_statement_index(node, entry_statements, owner):
    """Which top-level statement of `entry_statements` this node sits inside
    (by identity), or None if it is not directly part of one of them.

    `.parent` does not stop at function boundaries, so a plain upward walk
    would happily match a write inside a nested closure against whatever
    outer statement happens to textually contain that closure -- e.g. a
    `return function () { ...write here... };` statement, attributing the
    write to "the position where the closure is returned." That is exactly
    backwards: declaring or returning a closure does not invoke it, so a
    write that only happens when that closure is later *called* has no
    business being treated as part of `owner`'s own unconditional prefix.
    The walk must stop the instant it crosses into a different function than
    `owner`, not just whenever it happens to find *some* nid match.
    """
    statement_ids = {statement.nid: i for i, statement in enumerate(entry_statements)}
    current = node
    #while current is not None:
    while current is not None and current is not owner:
        index = statement_ids.get(current.nid)
        if index is not None:
            return index
        if current.kind in FUNCTION_KINDS:
            return None
        current = current.parent
    return None


def _contains_call(node):
    """Whether a CallExpression or NewExpression runs during this statement.
    Defining a function does not run its body, so don't look inside one."""
    stack = [node]
    while stack:
        current = stack.pop()
        if current.kind in FUNCTION_KINDS:
            continue
        if current.kind in ('CallExpression', 'NewExpression'):
            return True
        for _slot, child, _role in current.iter_children():
            stack.append(child)
    return False


def _plain_write_value(write, facts):
    """The concrete value a plain (non-compound) write assigns, or None if
    the write is not plain, has no resolvable value, or isn't concrete.

    Deliberately does not handle `+=` and friends: for those, evaluating the
    write's own right-hand side needs the binding's *current* value, which
    for a non-promotable binding reads back through `Solver.pinned` -- empty
    on the very solve these facts came from. Rather than guess, such writes
    are simply not settled by this pass.
    """
    parent = write.parent
    if parent is None:
        return None
    if parent.kind == 'VariableDeclarator' and parent.get('id') is write:
        init = parent.get('init')
        if init is None:
            return None
        value = facts.value_of(init)
    elif parent.kind == 'AssignmentExpression' and parent.get('left') is write \
            and parent.attr('operator', '=') == '=':
        value = facts.value_of(parent.get('right'))
    else:
        return None
    if value in V.LATTICE_MARKERS or not V.is_concrete(value):
        return None
    return value


def pin_settled_escaping_bindings(index, results):
    """Escaping bindings that are fully written, unconditionally, before their
    owner function could possibly call anything able to observe them.

    A closure sees whatever a captured variable holds *when the closure
    actually runs*, not a snapshot from when it was created -- declaring a
    `function`, or assigning a function expression, does not invoke it. So if
    every write to an escaping binding is a top-level statement of its
    owner's entry block (no branch, no loop, nothing conditional about
    whether or in what order they run), none of those writes happen from
    inside a nested function, and no call or `new` expression appears
    anywhere in the entry block before the last of those writes, then nothing
    could have run yet that might read this binding: the first thing capable
    of invoking anything hasn't executed. From that point on nothing writes
    it again, so it is safe to treat as a constant for the rest of the
    owner's lifetime, including from the closures that capture it.

    This is a soundness trade, not a strengthening of anything already
    correct: it assumes the owner function is not reentered and its closures
    are not invoked out of the order the source implies (recursion, a
    callback firing synchronously during setup, etc). That is the same bet
    most obfuscated-code deobfuscation already makes about "declare some
    constants, then define helpers that close over them" -- it is what an
    obfuscator emits for that pattern mechanically, not a general truth about
    JS, so treat it as a heuristic to widen deliberately rather than assume
    it covers every escaping binding.

    Only plain assignments/declarations are considered (see
    `_plain_write_value`); a binding whose settling write is compound, or
    whose value isn't concrete in `results`, is left alone. Bindings declared
    at Program (module/script) scope aren't covered -- `Binding.scope.function`
    is `None` there, and there is no single owner CFG to check against.
    """
    pinned = {}
    for binding in index.all_bindings():
        if binding.promotable or not binding.escapes:
            continue
        if not binding.writes or binding.kind not in ('var', 'let', 'param'):
            continue

        owner = binding.scope.function
        if owner is None:
            continue
        entry = results.get(owner.nid)
        if entry is None:
            continue
        facts, cfg = entry
        entry_statements = cfg.entry.statements

        positions = []
        settled = True
        for write in binding.writes:
            #position = _entry_statement_index(write, entry_statements)
            position = _entry_statement_index(write, entry_statements, owner)
            if position is None:
                settled = False
                break
            positions.append((position, write))
        if not settled:
            continue

        positions.sort(key=lambda item: item[0])
        last_position, last_write = positions[-1]

        if any(_contains_call(statement)
               for statement in entry_statements[:last_position + 1]):
            continue

        value = _plain_write_value(last_write, facts)
        if value is None:
            continue

        pinned[binding] = value
    return pinned
