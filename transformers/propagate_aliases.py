"""
propagate_aliases -- copy propagation for `v = s` where both sides are variables.

The solver only propagates *values*. `v39 = K0u1z` has no concrete value to
propagate (K0u1z is an escaping, program-scope binding), so `v39[0]` is left
reading through the copy. This pass rewrites reads of `v` to read `s` directly,
which exposes `K0u1z[0]` to fold_arrays and leaves the copy dead for
fold_constants to collect.

Replacing a read of `v` with `s` is sound for one use `u` when all of these hold:

  1. v has exactly one effective write, the copy `v = s` / `var v = s`.
  2. The copy dominates `u`, in the same function. Every path to `u` passes the
     copy, so v holds s's value from the copy's last execution.
  3. s cannot have changed since that execution. One of:
       a. s is a parameter or function declaration that is never reassigned;
       b. s is local to the copy's function (so recursion gets a fresh s) and
          *every* write to s dominates the copy. A write between the copy and
          `u` would need a path copy -> write -> u that avoids the copy, which
          dominance rules out;
       c. s is declared in an enclosing function (or Program), has a single
          write, and that write is a top-level statement of its owner with no
          call before it -- so nothing could have run the copy first. This is
          the same bet `pin_settled_escaping_bindings` makes.
  4. The name `s` at `u` resolves to the same binding. A same-named declaration
     anywhere between `u` and s's scope (a parameter, a catch parameter, a
     nested `let`) would capture the new reference. Checked syntactically and
     deliberately over-approximated: a declaration of that name anywhere in the
     enclosing function counts as shadowing.

Uses that fail any condition are left alone; the others are still rewritten.
Chains (`a = b; b = c`) take one link per run -- use run_until_stable. Copies are
removed only after all rewrites, and never while a rewrite has just made them live.
"""

from jsir import node as make, walk
from jsir.flow import build_cfg, dominators
from jsir.schema import FUNCTION_KINDS
from jsir.ccp import _entry_statement_index, _contains_call

# Shared helpers; adjust to the package layout.
from transformers.fold_arrays import _effective_writes, _trackable, _in_target_position


# ---------------------------------------------------------------------------
# Regions and ordering
# ---------------------------------------------------------------------------

def _region_of(node):
    """The function a node executes in, or None for Program."""
    return node.enclosing(*FUNCTION_KINDS)


def _locate(cfg, node, owner):
    """(block, emitted statement) holding `node`, or None."""
    current = node
    while current is not None and current is not owner:
        block = cfg.block_of_statement.get(current.nid)
        if block is not None:
            return block, current
        current = current.parent
    return None


class _Flow:
    def __init__(self, tree):
        self.root = tree.root
        self._cache = {}

    def _owner(self, region):
        return region if region is not None else self.root

    def _analysis(self, region):
        owner = self._owner(region)
        if owner.nid not in self._cache:
            cfg = build_cfg(owner)
            self._cache[owner.nid] = (cfg, dominators(cfg))
        return self._cache[owner.nid]

    def before(self, a, b, region):
        """Does `a` run strictly before `b` on every path that reaches `b`?
        False for two nodes in one emitted statement: order inside an
        expression is not modelled."""
        cfg, doms = self._analysis(region)
        owner = self._owner(region)
        left = _locate(cfg, a, owner)
        right = _locate(cfg, b, owner)
        if left is None or right is None:
            return False
        (block_a, stmt_a), (block_b, stmt_b) = left, right
        if stmt_a is stmt_b:
            return False
        if block_a is block_b:
            order = block_a.statements
            ia = next((i for i, s in enumerate(order) if s is stmt_a), None)
            ib = next((i for i, s in enumerate(order) if s is stmt_b), None)
            return ia is not None and ib is not None and ia < ib
        return doms.dominates(block_a.bid, block_b.bid)

    def settled(self, write, owner_function):
        """`write` is a top-level statement of its owner and no call or `new`
        runs at or before it, so no function could have executed yet."""
        owner = self._owner(owner_function)
        cfg, _doms = self._analysis(owner_function)
        entry_statements = cfg.entry.statements
        position = _entry_statement_index(write, entry_statements, owner)
        if position is None:
            return False
        return not any(_contains_call(s) for s in entry_statements[:position + 1])


# ---------------------------------------------------------------------------
# Name resolution at the use site (condition 4)
# ---------------------------------------------------------------------------

def _pattern_ids(pattern):
    if pattern is None:
        return
    kind = pattern.kind
    if kind == 'Identifier':
        yield pattern
    elif kind == 'AssignmentPattern':
        yield from _pattern_ids(pattern.get('left'))
    elif kind == 'RestElement':
        yield from _pattern_ids(pattern.get('argument'))
    elif kind == 'ArrayPattern':
        for element in pattern.children_in('elements'):
            yield from _pattern_ids(element)
    elif kind == 'ObjectPattern':
        for prop in pattern.children_in('properties'):
            if prop.kind == 'Property':
                yield from _pattern_ids(prop.get('value'))
            else:
                yield from _pattern_ids(prop)


def _declared_here(scope_node, name):
    """Identifiers declaring `name` anywhere in this function (or Program),
    not looking inside nested functions. Block-level `let`/`const`/catch
    parameters are counted as function-level: over-approximate on purpose."""
    found = []

    def add(ids):
        found.extend(i for i in ids if i.attr('name') == name)

    if scope_node.kind in FUNCTION_KINDS:
        for param in scope_node.children_in('params'):
            add(_pattern_ids(param))
        if scope_node.kind == 'FunctionExpression' and scope_node.get('id') is not None:
            add([scope_node.get('id')])
        body = scope_node.get('body')
        stack = [body] if body is not None else []
    else:
        stack = list(scope_node.children_in('body'))

    while stack:
        current = stack.pop()
        kind = current.kind
        if kind == 'VariableDeclaration':
            for declarator in current.children_in('declarations'):
                add(_pattern_ids(declarator.get('id')))
        elif kind in ('FunctionDeclaration', 'ClassDeclaration'):
            if current.get('id') is not None:
                add([current.get('id')])
        elif kind == 'CatchClause':
            add(_pattern_ids(current.get('param')))
        if kind in FUNCTION_KINDS:
            continue
        for _slot, child, _role in current.iter_children():
            stack.append(child)
    return found


def _resolves_to(index, name, use, source):
    """Would a fresh reference to `name` at `use` bind to `source`?"""
    if name == 'arguments':
        return False
    for scope_node in use.ancestors():
        if scope_node.kind not in FUNCTION_KINDS and scope_node.kind != 'Program':
            continue
        found = _declared_here(scope_node, name)
        if found:
            return all(index.binding_of(ident) is source for ident in found)
    return False


# ---------------------------------------------------------------------------
# Source stability (condition 3)
# ---------------------------------------------------------------------------

def _mentions_arguments(function):
    return any(n.kind == 'Identifier' and n.attr('name') == 'arguments'
               for n in walk(function))


def _source_stable(flow, source, copy_node, region):
    kind = source.kind
    scope = getattr(source, 'scope', None)
    if getattr(source, 'implicit_global', False) or \
            (scope is not None and getattr(scope, 'has_dynamic_scope', False)):
        return False

    if kind == 'param':
        for write in source.writes:
            if write.parent is None or write.parent.kind not in FUNCTION_KINDS:
                return False                     # reassigned somewhere
        owner = getattr(scope, 'function', None)
        return owner is not None and not _mentions_arguments(owner)

    if kind == 'function':
        for write in source.writes:
            parent = write.parent
            if parent is None or parent.kind != 'FunctionDeclaration' \
                    or parent.get('id') is not write:
                return False
        return True

    if kind not in ('var', 'let', 'const'):
        return False
    writes = _effective_writes(source)
    if not writes:
        return False

    owner = getattr(scope, 'function', None)     # None: Program scope
    if owner is region:
        return all(_region_of(w) is region and flow.before(w, copy_node, region)
                   for w in writes)

    if len(writes) != 1:
        return False
    if owner is not None and not any(a is owner for a in (region.ancestors() if region else ())):
        return False
    return flow.settled(writes[0], owner)


# ---------------------------------------------------------------------------
# Planning and applying
# ---------------------------------------------------------------------------

class _Plan:
    def __init__(self, alias, source, name, copy_node, uses):
        self.alias = alias
        self.source = source
        self.name = name
        self.copy_node = copy_node      # VariableDeclarator or AssignmentExpression
        self.uses = uses


def _copy_of(alias):
    """(copy node, the Identifier it reads) if `alias` is written exactly once,
    by a copy of another variable."""
    writes = _effective_writes(alias)
    if len(writes) != 1:
        return None
    write = writes[0]
    parent = write.parent
    if parent is None:
        return None
    if parent.kind == 'VariableDeclarator' and parent.get('id') is write:
        source = parent.get('init')
    elif parent.kind == 'AssignmentExpression' and parent.get('left') is write \
            and parent.attr('operator', '=') == '=':
        source = parent.get('right')
    else:
        return None
    if source is None or source.kind != 'Identifier':
        return None
    return parent, source


def _plan_for(index, flow, alias):
    if not _trackable(alias):
        return None
    found = _copy_of(alias)
    if found is None:
        return None
    copy_node, source_ident = found
    source = index.binding_of(source_ident)
    if source is None or source is alias:
        return None
    region = _region_of(copy_node)

    # A captured v is shared between activations; with a region-local s a
    # recursive call could overwrite it between the copy and the use.
    scope_owner = getattr(getattr(source, 'scope', None), 'function', None)
    if scope_owner is region and getattr(alias.scope, 'function', None) is not region:
        return None

    if not _source_stable(flow, source, copy_node, region):
        return None

    name = source_ident.attr('name')
    uses = []
    for use in alias.reads:
        if not use.live or use is source_ident or _in_target_position(use):
            continue
        if _region_of(use) is not region:
            continue
        if not flow.before(copy_node, use, region):
            continue
        if not _resolves_to(index, name, use, source):
            continue
        uses.append(use)
    return _Plan(alias, source, name, copy_node, uses) if uses else None


def _rewrite(tree, plan, added):
    rewritten = 0
    for use in plan.uses:
        if use.live:
            tree.replace(use, make('Identifier', name=plan.name))
            added[plan.source] = added.get(plan.source, 0) + 1
            rewritten += 1
    return rewritten


def _drop_dead_copy(tree, plan, added):
    """Remove the copy once nothing reads the alias. `added` matters: in a chain
    (`a = p; b = a; use(b)`) the plan for `b` rewrites `use(b)` into a *new* read
    of `a`, after `a`'s own reads looked dead. Removing `a = p` then would leave
    that read pointing at a variable nothing assigns."""
    if any(r.live for r in plan.alias.reads) or added.get(plan.alias, 0):
        return False
    if not plan.copy_node.live:
        return False
    if plan.copy_node.kind == 'VariableDeclarator':
        tree.set_slot(plan.copy_node, 'init', None)
        return True
    else:
        statement = plan.copy_node.parent
        if statement is not None and statement.kind == 'ExpressionStatement':
            tree.remove(statement)
            return True
    return False


def propagate_aliases(session) -> None:
    print("--- Applying Transformer: propagate_aliases ---")
    index = session.scopes
    flow = _Flow(session.tree)
    plans = []
    for binding in index.all_bindings():
        plan = _plan_for(index, flow, binding)
        if plan is not None:
            plans.append(plan)

    added = {}              # source binding -> references this run created
    rewritten_reads = 0
    aliases_propagated = 0

    for plan in plans:
        count = _rewrite(session.tree, plan, added)
        if count > 0:
            rewritten_reads += count
            aliases_propagated += 1

    dropped_copies = 0
    for plan in plans:      # only after every rewrite: see _drop_dead_copy
        if _drop_dead_copy(session.tree, plan, added):
            dropped_copies += 1

    print(
        f" |-> propagated {aliases_propagated} alias(es) "
        f"across {rewritten_reads} read site(s), dropped {dropped_copies} dead copy node(s)."
    )
