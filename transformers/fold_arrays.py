"""
fold_arrays -- propagate constant elements out of array literals that are only read.

The solver evaluates `ArrayExpression` to UNKNOWN, so nothing downstream can fold
`K[1]`. Rather than teach the lattice about heap objects (identity, aliasing,
mutation), this pass proves a narrower thing syntactically and then rewrites:

    A binding holds an array if it has exactly one effective write, that write
    is an array literal, and every reference to it -- through any chain of
    single-write aliases (`v = K`) -- is one of:
        K[<const int>]      a read of one element        -> inlined if constant
        K[<anything else>]  a dynamic read                -> pins the array
        K.length, K.join()  whole-array reads             -> pins the array
    Anything else (passed to a call, returned, stored, `K[0] = x`, `K.push(..)`,
    `delete K[0]`, `for (K[0] in ..)`, ...) could mutate or leak the array, so the
    binding is left alone entirely.

For such a binding:
  * `K[n]` is replaced by element n's value when the solver says it is a
    primitive constant;
  * if some elements stay referenced (non-constant ones) and *every* read is a
    static index, the elements nobody needs are removed and the surviving
    references renumbered -- `K = [a, f, b]; K[1]` becomes `K = [f]; K[0]`;
  * alias writes whose reads were all inlined are removed; the array's own
    write and declarators are left for fold_constants' dead-code pass.

Run it before fold_constants, under run_until_stable, so the literals it
produces fold and the emptied bindings get collected.

Assumptions (the same bets pin_settled_escaping_bindings already makes):
  * The input program runs. `K[n]` throws if K is not yet the array, so a read
    that executes before the write cannot exist in a working program; the only
    thing inlining changes is a read that would have thrown inside try/catch.
  * Program-scope `var`s are not reached through the global object by name
    (`window['K']`, `globalThis[expr]`). Flip TRUST_PROGRAM_SCOPE to disable.
"""

import math

from jsir import node as make, values as V
from jsir.schema import slots_of, TARGET

# Adjust to the package layout; these are the helpers fold_constants already has.
from transformers.fold_constants import _value_to_node, _has_effects, _result_is_discarded

TRUST_PROGRAM_SCOPE = True

# Methods that read an array without changing it or leaking an alias to it.
_PURE_METHODS = frozenset(('join', 'slice', 'indexOf', 'lastIndexOf', 'includes',
                           'concat', 'at', 'toString'))

_INDEXED = 'indexed'
_PIN = 'pin'
_ALIAS = 'alias'
_TAINT = 'taint'


# ---------------------------------------------------------------------------
# Small predicates
# ---------------------------------------------------------------------------

def _array_index(value):
    """The array index a key value denotes, or None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if value == int(value) and value >= 0:
            return int(value)
        return None
    if isinstance(value, str) and value.isascii() and value.isdigit() \
            and str(int(value)) == value:
        return int(value)
    return None


def _inlineable(value):
    """Only immutable primitives: inlining an object would change identity."""
    if value is None:
        return False
    if value is V.NULL or value is V.UNDEF:
        return True
    return isinstance(value, (str, bool, int, float))


def _effective_writes(binding):
    """Writes that actually store a value. `var x;` stores nothing."""
    out = []
    for write in binding.writes:
        parent = write.parent
        if parent is not None and parent.kind == 'VariableDeclarator' \
                and parent.get('id') is write and parent.get('init') is None:
            declaration = parent.parent
            head = declaration.parent if declaration is not None else None
            if not (head is not None and head.kind in ('ForInStatement', 'ForOfStatement')
                    and head.get('left') is declaration):
                continue
        out.append(write)
    return out


def _trackable(binding):
    if binding is None or binding.kind not in ('var', 'let', 'const'):
        return False
    if getattr(binding, 'implicit_global', False):
        return False
    scope = getattr(binding, 'scope', None)
    if scope is not None and getattr(scope, 'has_dynamic_scope', False):
        return False
    if not TRUST_PROGRAM_SCOPE and getattr(scope, 'function', None) is None:
        return False
    return True


def _in_target_position(node):
    """Is `node` the thing being written, rather than read?"""
    parent = node.parent
    if parent is None:
        return False
    for slot, _arity, role in slots_of(parent.kind):
        if slot == node.parent_slot and role == TARGET:
            return True
    if parent.kind == 'UnaryExpression' and parent.attr('operator') == 'delete':
        return True
    # `({ a: K[0] } = x)`: Property.value is not a TARGET slot, its pattern is.
    if parent.kind == 'Property' and parent.get('value') is node \
            and parent.parent is not None and parent.parent.kind == 'ObjectPattern':
        return True
    return False


def _static_key(session, member):
    prop = member.get('property')
    if prop is None:
        return None
    if not member.attr('computed', False):
        return prop.attr('name') if prop.kind == 'Identifier' else None
    if prop.kind == 'Literal':
        return prop.attr('value')
    return session.constant_of(prop)


def _array_literal_of(binding):
    """The ArrayExpression this binding is set to, if that is its only write."""
    writes = _effective_writes(binding)
    if len(writes) != 1:
        return None
    write = writes[0]
    parent = write.parent
    if parent is None:
        return None
    if parent.kind == 'VariableDeclarator' and parent.get('id') is write:
        value = parent.get('init')
    elif parent.kind == 'AssignmentExpression' and parent.get('left') is write \
            and parent.attr('operator', '=') == '=' and _result_is_discarded(parent):
        value = parent.get('right')
    else:
        return None
    if value is not None and value.kind == 'ArrayExpression':
        return value
    return None


# ---------------------------------------------------------------------------
# Classification of one reference
# ---------------------------------------------------------------------------

def _classify(session, index, ident):
    """-> (kind, payload...) describing how `ident` (a read of the array) uses it."""
    parent = ident.parent
    if parent is None:
        return (_TAINT,)

    if parent.kind == 'MemberExpression' and parent.get('object') is ident:
        member = parent
        grand = member.parent
        if grand is None or _in_target_position(member):
            return (_TAINT,)
        key = _static_key(session, member)
        optional = member.attr('optional', False)

        if grand.kind in ('CallExpression', 'NewExpression') and grand.get('callee') is member:
            if grand.kind == 'CallExpression' and not optional \
                    and isinstance(key, str) and key in _PURE_METHODS:
                return (_PIN,)
            return (_TAINT,)
        if grand.kind == 'TaggedTemplateExpression' and grand.get('tag') is member:
            return (_TAINT,)

        n = _array_index(key)
        if n is not None and not optional:
            return (_INDEXED, member, n)
        if key is None or key == 'length':
            return (_PIN,)          # dynamic key (assumed an index), or `.length`
        return (_TAINT,)            # `K.push` as a value, `K.foo`, ...

    # Alias: `var v = K` / `v = K` where v is written exactly once, here.
    target = None
    if parent.kind == 'VariableDeclarator' and parent.get('init') is ident:
        target = parent.get('id')
    elif parent.kind == 'AssignmentExpression' and parent.get('right') is ident \
            and parent.attr('operator', '=') == '=' and _result_is_discarded(parent):
        target = parent.get('left')
    if target is not None and target.kind == 'Identifier':
        alias = index.binding_of(target)
        if _trackable(alias) and _effective_writes(alias) == [target]:
            return (_ALIAS, alias, parent)
    return (_TAINT,)


# ---------------------------------------------------------------------------
# Planning (read-only: every analysis query happens before the first edit)
# ---------------------------------------------------------------------------

class _Plan:
    def __init__(self, array, elements, values):
        self.array = array
        self.elements = elements
        self.values = values            # per element: primitive value or None
        self.indexed = []               # (member, n)
        self.aliases = []               # (alias_binding, assignment_node)
        self.pinned = False


def _plan_for(session, index, root):
    if not _trackable(root):
        return None
    array = _array_literal_of(root)
    if array is None:
        return None
    elements = list(array.children_in('elements'))
    if any(e is None or e.kind == 'SpreadElement' for e in elements):
        return None                     # holes / spreads: indices not knowable
    plan = _Plan(array, elements, [None] * len(elements))

    seen = {root}
    queue = [root]
    while queue:
        binding = queue.pop()
        for ident in binding.reads:
            if not ident.live:
                continue
            result = _classify(session, index, ident)
            kind = result[0]
            if kind == _TAINT:
                return None
            if kind == _PIN:
                plan.pinned = True
            elif kind == _INDEXED:
                _kind, member, n = result
                if n >= len(elements):
                    plan.pinned = True  # reads `undefined`; leave it alone
                else:
                    plan.indexed.append((member, n))
            else:
                _kind, alias, assignment = result
                plan.aliases.append((alias, assignment))
                if alias not in seen:
                    seen.add(alias)
                    queue.append(alias)

    if not plan.indexed:
        return None
    for i, element in enumerate(elements):
        value = session.constant_of(element)
        plan.values[i] = value if _inlineable(value) else None
    return plan


# ---------------------------------------------------------------------------
# Applying
# ---------------------------------------------------------------------------

def _apply(tree, plan):
    remaining = {}                      # element index -> members still reading it
    for member, n in plan.indexed:
        if not member.live:
            continue
        value = plan.values[n]
        if value is None:
            remaining.setdefault(n, []).append(member)
            continue
        replacement = _value_to_node(value)
        if replacement is None:
            remaining.setdefault(n, []).append(member)
            continue
        tree.replace(member, replacement)

    # Partial propagation: drop elements nothing needs, renumber the rest.
    if not plan.pinned and remaining:
        kept = [i for i, element in enumerate(plan.elements)
                if i in remaining or _has_effects(element)]
        dropped = [i for i in range(len(plan.elements)) if i not in set(kept)]
        if dropped:
            mapping = {old: new for new, old in enumerate(kept)}
            for old, members in remaining.items():
                if mapping[old] == old:
                    continue
                for member in members:
                    if member.live:
                        tree.set_slot(member, 'property',
                                      make('Literal', value=float(mapping[old])))
            for i in dropped:
                if plan.elements[i].live:
                    tree.remove(plan.elements[i])

    # Alias writes whose every read was inlined are now pure noise.
    for alias, assignment in plan.aliases:
        if not assignment.live or any(r.live for r in alias.reads):
            continue
        if assignment.kind == 'VariableDeclarator':
            tree.set_slot(assignment, 'init', None)
        else:
            statement = assignment.parent
            if statement is not None and statement.kind == 'ExpressionStatement':
                tree.remove(statement)


def fold_arrays(session) -> None:
    print("--- Applying Transformer: fold_arrays ---")
    index = session.scopes
    plans = []
    for binding in index.all_bindings():
        plan = _plan_for(session, index, binding)
        if plan is not None:
            plans.append(plan)
    for plan in plans:
        _apply(session.tree, plan)
