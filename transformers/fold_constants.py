"""
fold_constants — constant folding and dead code elimination.

CCP has computed a per-block environment joined correctly at every
merge. A node gets folded only when the solver recorded a concrete value for it;
if the solver reached a join with two different values, the node recorded UNKNOWN
and is left alone.

Dead assignment elimination: a write whose binding has no reads after all
replacements is removed. Dead declarator elimination: a declarator whose binding
has no reads and no remaining writes is removed.
"""

import math
import re

from jsir import Session, walk, walk_post, clone, node as make, values as V
from jsir.values import is_concrete
from jsir.schema import FUNCTION_KINDS

_IDENTIFIER_RE = re.compile(r'^[a-zA-Z_$][a-zA-Z0-9_$]*$')

_EFFECT_KINDS = frozenset((
    'AssignmentExpression', 'UpdateExpression', 'CallExpression',
    'NewExpression', 'TaggedTemplateExpression', 'YieldExpression',
    'AwaitExpression',
))

def _has_effects(expr):
    """Conservative: any write, call, or delete anywhere in `expr`,
    not counting the bodies of nested functions."""
    if expr is None or expr.kind in FUNCTION_KINDS:
        return False
    for n in walk(expr, enter_functions=False):
        if n.kind in _EFFECT_KINDS:
            return True
        if n.kind == 'UnaryExpression' and n.attr('operator') == 'delete':
            return True
    return False

def _value_to_node(value):
    """Build an IR node for a concrete value, or None if it cannot be expressed
    as a literal. Callers must use an Identifier for UNDEF and NaN."""
    if not is_concrete(value):
        return None
    if value is V.UNDEF:
        return make('Identifier', name='undefined')
    if value is V.NULL:
        return make('Literal', value=V.NULL)
    if isinstance(value, float):
        if math.isnan(value):
            return make('Identifier', name='NaN')
        if math.isinf(value):
            # 1/0 is the only literal-safe representation of Infinity that
            # cannot be shadowed by a local variable.
            one = make('Literal', value=1.0)
            zero = make('Literal', value=0.0)
            expr = make('BinaryExpression', operator='/', left=one, right=zero)
            return make('UnaryExpression', operator='-', prefix=True, argument=expr) \
                if value < 0 else expr
        if value < 0 or (value == 0.0 and math.copysign(1.0, value) < 0):
            magnitude = -value
            if magnitude == int(magnitude) and magnitude < 1e21:
                magnitude = int(magnitude)
            return make('UnaryExpression', operator='-', prefix=True,
                        argument=make('Literal', value=float(magnitude)))
        if value == int(value) and value < 1e21:
            #return make('Literal', value=int(value))
            return make('Literal', value=float(value))
        return make('Literal', value=value)
    if isinstance(value, V.JSRegExp):
        return make('Literal', value=value)
    # bool, str
    return make('Literal', value=value)


def _same_node(a, b):
    """Would replacing a with b change the source? Used to skip no-op rewrites."""
    if a.kind != b.kind:
        return False
    if a.attrs != b.attrs:
        return False

    # Compare slots/children recursively for complex folded nodes like -5 or -1/0
    a_children = list(a.iter_children())
    b_children = list(b.iter_children())
    if len(a_children) != len(b_children):
        return False

    for (slot_a, child_a, _), (slot_b, child_b, _) in zip(a_children, b_children):
        if slot_a != slot_b or not _same_node(child_a, child_b):
            return False

    return True


def _is_for_init_or_update(for_stmt, node):
    """`node` fills the init or update slot of `for_stmt`. Neither slot's value
    is consumed (only `test` is), so an expression there is run for effect."""
    return for_stmt.get('update') is node or for_stmt.get('init') is node


def _reads_binding(expr, binding):
    """Does `expr` contain a read of `binding`? Used before salvaging
    `x = f(x)` -> `f(x)`: the surviving `x` is then a real read."""
    if expr is None:
        return False
    return any(n in binding.reads for n in walk(expr))


def _result_is_discarded(expr):
    """True if `expr` (an UpdateExpression or a compound AssignmentExpression)
    is evaluated purely for its side effect: nothing reads the value it
    produces. `x++;`, `for (x = 0; ...; x++)` (init and update) qualify; `y = x++` does not,
    because x's pre-increment value -- the same value the read half of the
    update computes -- flows into y."""
    node = expr
    while True:
        parent = node.parent
        if parent is None:
            return True
        if parent.kind == 'ExpressionStatement':
            return True
        if parent.kind == 'ForStatement' and _is_for_init_or_update(parent, node):
            return True
        if parent.kind == 'SequenceExpression':
            exprs = parent.children_in('expressions')
            if exprs and node is not exprs[-1]:
                return True
            node = parent   # last element: keep climbing past the sequence itself
            continue
        return False


def _is_write_read(node, binding):
    """Returns True if `node` is a read that only exists as part of a write to
    `binding`, where that write's own result is never itself consumed. Such a
    node is not a real downstream user of the binding's value -- it only
    exists to compute the binding's *next* value -- so it should not keep an
    otherwise-unread binding alive.

    Only `x++`/`x--` and compound assignment (`x += 1`) route through the
    result-discarded check below: those are the only writes that also appear
    in `binding.reads` (see binding.py's UPDATE usage), so they are the only
    case where `node in binding.writes` can be true for a node that is being
    asked about as one of `binding.reads`. A plain write target (`x = 5`) is
    never itself a read and never reaches this function via `surviving_reads`.
    """
    if node in binding.writes:
        curr = node.parent
        if curr is not None and curr.kind == 'UpdateExpression' and curr.get('argument') is node:
            return _result_is_discarded(curr)
        if curr is not None and curr.kind == 'AssignmentExpression' and \
                curr.attr('operator', '=') != '=' and curr.get('left') is node:
            return _result_is_discarded(curr)
        return True
    curr = node.parent
    while curr is not None:
        if curr.kind == 'AssignmentExpression':
            if curr.get('left') in binding.writes:
                return True
        elif curr.kind == 'UpdateExpression':
            if curr.get('argument') in binding.writes:
                return True
        elif curr.kind in ('FunctionDeclaration', 'FunctionExpression', 'ArrowFunctionExpression', 'Program'):
            break
        curr = curr.parent
    return False


def _removable_declarator(declarator):
    target = declarator.get('id')
    return (target is not None and target.kind == 'Identifier'
            and not _has_effects(declarator.get('init')))


def fold_constants(session: Session) -> None:
    print("--- Applying Transformer: fold_constants ---")

    tree = session.tree
    all_facts = session.facts

    # -- Pass 1: fold -------------------------------------------------------

    foldable = []
    for node in walk(tree.root):
        kind = node.kind

        # Do not fold assignment or update expressions directly into literals;
        # let Pass 3 handle dead assignment elimination.
        if kind in ('AssignmentExpression', 'UpdateExpression', 'SequenceExpression'):
            continue

        parent = node.parent
        if parent is not None:
            pk = parent.kind
            if pk == 'AssignmentExpression' and parent.get('left') is node:
                continue
            if pk == 'UpdateExpression' and parent.get('argument') is node:
                continue
            if pk in ('VariableDeclarator', 'FunctionDeclaration',
                      'FunctionExpression', 'ArrowFunctionExpression') and \
                    parent.get('id') is node:
                continue
            if pk == 'Property' and parent.get('key') is node and \
                    not parent.attr('computed', False):
                continue
            if pk == 'MemberExpression' and parent.get('property') is node and \
                    not parent.attr('computed', False):
                continue

        facts_entry = all_facts.get(_owner_nid(node, session))
        if facts_entry is None:
            continue
        facts = facts_entry[0]
        value = facts.constant_of(node)
        if value is None:
            continue

        replacement = _value_to_node(value)
        if replacement is None or _same_node(node, replacement):
            continue
        foldable.append((node, replacement))

    folded_count = 0
    for node, replacement in foldable:
        if node.live:
            tree.replace(node, replacement)
            folded_count += 1

    # -- Pass 2: computed string key → static member ------------------------

    computed_members = []
    for node in walk(tree.root, kinds={'MemberExpression'}):
        if not node.attr('computed', False):
            continue
        prop = node.get('property')
        if prop is None or prop.kind != 'Literal':
            continue
        key = prop.attr('value')
        if not isinstance(key, str) or not _IDENTIFIER_RE.match(key):
            continue

        computed_members.append((node, key))

    computed_count = 0
    for node, key in computed_members:
        if not node.live:
            continue
        new_prop = make('Identifier', name=key)
        tree.set_slot(node, 'property', new_prop)
        node.attrs['computed'] = False
        computed_count += 1

    print(
        f"fold_constants: folded {folded_count} constant expression(s), "
        f"converted {computed_count} computed property access(es) to static."
    )

    # -- Pass 3: dead code elimination --------------------------------------
def dead_code_elimination(session: Session) -> None:
    print("--- Applying Transformer: dead_code_elimination ---")
    tree = session.tree

    index = session.scopes

    dead_statements = []
    dead_declarators = []
    dead_updates = []
    salvage = []

    keep = getattr(session, 'dce_keep', frozenset())
    root = getattr(session, 'runtime_root', None)
    runtime_scope = None
    if root:
        runtime_scope = index.scope_of(root).var_scope()

    for binding in index.all_bindings():
        if runtime_scope is not None and binding.name in keep and binding.scope is runtime_scope:
            continue
        surviving_reads = [
            r for r in binding.reads
            if r.live and not _is_write_read(r, binding)
        ]
        if surviving_reads or binding.escapes:
            continue
        if binding.kind in ('param', 'function', 'class', 'import', 'catch'):
            continue

        # The declaration may only go if EVERY write to the binding goes too.
        # Otherwise the surviving write turns into an implicit global (or a
        # ReferenceError in strict mode). `keep_declaration` records that some
        # write was left in place or still reads the binding.
        keep_declaration = False
        binding_declarators = []

        declaration = binding.declaration
        if declaration is not None and declaration.kind == 'VariableDeclarator' \
                and declaration.live and _removable_declarator(declaration):
            binding_declarators.append(declaration)

        for write_ident in list(binding.writes):
            if not write_ident.live:
                continue
            decl = write_ident.parent
            if decl is None:
                keep_declaration = True
                continue
            if decl.kind == 'VariableDeclarator':
                if _removable_declarator(decl):
                    binding_declarators.append(decl)
                # else: declarator stays with its init, so it stays declared.
            elif decl.kind == 'AssignmentExpression' and decl.attr('operator', '=') == '=' \
                    and decl.get('left') is write_ident:
                if not _result_is_discarded(decl):
                    keep_declaration = True
                    continue
                right = decl.get('right')
                if _has_effects(right):
                    salvage.append(decl)
                    if _reads_binding(right, binding):
                        keep_declaration = True   # `x = f(x)` -> `f(x)`
                    continue
                stmt = decl.parent
                if stmt is not None and stmt.kind == 'ExpressionStatement':
                    dead_statements.append(stmt)
                elif stmt is not None and stmt.kind == 'SequenceExpression':
                    dead_updates.append(decl)
                elif stmt is not None and stmt.kind == 'ForStatement' \
                        and _is_for_init_or_update(stmt, decl):
                    dead_updates.append(decl)
                else:
                    keep_declaration = True
            elif decl.kind in ('AssignmentExpression', 'UpdateExpression'):
                if decl.kind == 'AssignmentExpression' and _has_effects(decl.get('right')):
                    salvage.append(decl)
                    if _reads_binding(decl.get('right'), binding):
                        keep_declaration = True
                    continue
                # A compound assignment (`x += 1`) or update (`x++`/`x--`) whose
                # read half was excluded from surviving_reads above precisely
                # because _is_write_read proved its result is discarded: it is
                # itself dead code and must go along with the declarator.
                stmt = decl.parent
                if stmt is not None and stmt.kind == 'ExpressionStatement':
                    dead_statements.append(stmt)
                elif stmt is not None and stmt.kind == 'ForStatement' \
                        and _is_for_init_or_update(stmt, decl):
                    dead_updates.append(decl)
                elif stmt is not None and stmt.kind == 'SequenceExpression':
                    dead_updates.append(decl)
                else:
                    keep_declaration = True
            else:
                # Destructuring target, for-in/of head, etc.: not handled here.
                keep_declaration = True

        if not keep_declaration:
            dead_declarators.extend(binding_declarators)

    removed_statements = 0
    for stmt in dead_statements:
        if stmt.live:
            tree.remove(stmt)
            removed_statements += 1

    removed_updates = 0
    for update_node in dead_updates:
        if update_node.live:
            tree.remove(update_node)
            removed_updates += 1

    salvaged_assigns = 0
    for assign in salvage:
        if assign.live:
            tree.replace(assign, clone(assign.get('right')))
            salvaged_assigns += 1

    # Removing operands above can leave a SequenceExpression too thin to mean
    # anything: zero operands has no valid JS spelling at all (escodegen
    # crashes trying to print it -- an empty fragment reaching
    # ExpressionStatement's brace-check), and one operand is a comma
    # expression wrapped around a single value, which is legal but pointless.
    # Collapse both.
    collapsed_sequences = 0
    for seq in walk_post(tree.root):
        if seq.kind != 'SequenceExpression' or not seq.live:
            continue
        exprs = seq.children_in('expressions')
        if len(exprs) >= 2:
            continue
        if len(exprs) == 1:
            if seq.parent is not None:
                tree.replace(seq, clone(exprs[0]))
                collapsed_sequences += 1
            continue
        parent = seq.parent
        if parent is None:
            continue
        if parent.kind == 'ExpressionStatement':
            tree.remove(parent)
            collapsed_sequences += 1
        else:
            tree.remove(seq)
            collapsed_sequences += 1

    removed_declarators = 0
    for declarator in dead_declarators:
        if not declarator.live:
            continue
        decl_node = declarator.parent
        if decl_node is None or decl_node.kind != 'VariableDeclaration':
            continue

        grandparent = decl_node.parent
        if grandparent is not None and grandparent.kind in (
                'ForInStatement', 'ForOfStatement'):
            continue

        declarators = decl_node.children.get('declarations', [])
        if len(declarators) == 1:
            if decl_node.parent is not None:
                tree.remove(decl_node)
                removed_declarators += 1
        else:
            tree.remove(declarator)
            removed_declarators += 1

    print(
        f"dead_code_elimination: removed {removed_declarators} declarator(s), "
        f"{removed_statements} dead statement(s), {removed_updates} dead update(s), "
        f"salvaged {salvaged_assigns} assignment(s), collapsed {collapsed_sequences} sequence(s)."
    )


def _owner_nid(node, session):
    """The nid of the function body that owns this node, for facts lookup."""
    from jsir.schema import FUNCTION_KINDS
    current = node
    while current is not None:
        if current.kind in FUNCTION_KINDS or current.kind == 'Program':
            return current.nid
        current = current.parent
    return None
