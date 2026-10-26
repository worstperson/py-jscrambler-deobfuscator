"""
lift_sequence_expressions -- remove comma expressions wherever they can be
removed without changing evaluation order.

Where a SequenceExpression can live (ESTree), and what this pass does with it
==========================================================================

Hoisted to statements placed *before* the host statement
    ExpressionStatement.expression      a, b;                 -> a; b;
    ReturnStatement / ThrowStatement    return a, b;          -> a; return b;
    IfStatement.test                    if (a, b) ...         -> a; if (b) ...
    SwitchStatement.discriminant        switch (a, b) ...     -> a; switch (b) ...
    WithStatement.object
    ForIn/ForOf.right                   for (k in (a, b))     -> a; for (k in b)
    ForStatement.init (expression)      for (a, b;;)          -> a; for (b;;)
    VariableDeclarator.init             var x = (a, b);       -> a; var x = b;
                                        (declarations are split so that earlier
                                        declarators still run first)
    ClassDeclaration.superClass         class A extends (a, B) -> a; class A extends B
    ExportDefaultDeclaration            export default (a, b) -> a; export default b
    ArrowFunctionExpression.body        () => (a(), b)        -> () => { a(); return b; }

Hoisted out of *nested* positions (stacking)
    assignment RHS / compound RHS       x = y = (a, b)        -> a; x = y = b
    binary / unary / member object      (a, b).c, (a,b)+(c,d) -> ...
    call callee + arguments             f((a, b), (c, d))     -> a; c; f(b, d)   (if safe)
    new, tagged template, spread, array/object literals (incl. computed keys),
    template literals, await / yield, update targets, destructuring RHS,
    LHS of logical / left of ?:, test of ?:.

Loop tests (needs a loop rewrite, so it lives here)
    while ((a, b)) body             -> while (true) { a; if (!b) break; body }
    for (i; (a, b); u) body         -> for (i;; u) { a; if (!b) break; body }

Branches are NOT converted here. `a && (b, c)` and `c ? (a, b) : (d, e)` are
left in place (sequences inside conditional branches cannot be hoisted past
the condition). branches_to_control_flow.py converts those to if/else and is
meant to run late, followed by another run of this pass.

Deliberately left alone (cannot be hoisted without temporaries or changing
semantics):
    for-update, do-while test, switch-case tests, default parameter values,
    class field initialisers / computed class keys, the RHS of &&= ||= ??=,
    anything inside an optional chain, `delete (a, b)`, and any sequence whose
    evaluation would have to move past an earlier operand that the hoisted code
    might change (see "Safety model").

Semantic details that are handled
    * `(0, o.m)()` / `(0, eval)(x)` / tagged templates: the comma is what drops
      `this` / forces indirect eval, so the last element is kept wrapped in
      `(0, last)` and only the earlier effects are hoisted.
    * `typeof (a, undeclared)` and `delete (a, o.p)` are not rewritten.
    * Labels, `export`, and `for (init;;)` positions are never wrapped in a
      block; statements are emitted before the labeled / exported / loop
      statement instead.

Safety model
    A sequence nested inside a larger expression is only hoisted if every
    operand evaluated *before* it still means the same thing after the hoisted
    code runs earlier. Operands are checked against the effects of the hoisted
    code (identifier writes, member writes, opaque effects such as calls):
      - literals, `this`, function expressions: always fine
      - identifier: fine unless hoisted code assigns it, or hoisted code is
        opaque (calls) and the name is assigned anywhere in the program
        (name-based; never-assigned names are treated as constants)
      - member reads: fine only if hoisted code has no member writes / calls
      - anything with its own side effects (calls, updates, ...): blocks
    Options:
      aggressive=True        assume calls in hoisted code do not mutate variables
                             or properties read earlier in the same expression.
      is_local_stable(node)  hook, True if an Identifier is known not to be
                             changed by calls (e.g. from scope analysis).
"""

from collections import Counter

from jsir import Session, walk, node as make, clone
from jsir.schema import slots_of, MANY, KEY, LABEL, FUNCTION_KINDS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HOST_KINDS = frozenset((
    'ExpressionStatement', 'ReturnStatement', 'ThrowStatement', 'IfStatement',
    'SwitchStatement', 'WithStatement', 'ForInStatement', 'ForOfStatement',
    'ForStatement', 'WhileStatement', 'VariableDeclaration', 'ClassDeclaration',
    'ExportDefaultDeclaration', 'ArrowFunctionExpression',
))

_SLOT_HOSTS = {
    'IfStatement': 'test',
    'SwitchStatement': 'discriminant',
    'WithStatement': 'object',
    'ForInStatement': 'right',
    'ForOfStatement': 'right',
    'ClassDeclaration': 'superClass',
}

# Expressions that can be dropped when their value is unused.
_DROPPABLE = frozenset((
    'Literal', 'ThisExpression', 'FunctionExpression', 'ArrowFunctionExpression'))

# Leaves whose value cannot be changed by any side effect.
_LEAF_SAFE = frozenset((
    'Literal', 'ThisExpression', 'FunctionExpression', 'ArrowFunctionExpression',
    'TemplateElement', 'PrivateIdentifier', 'Super'))

# Pure composites: safe to move if all read-children are safe.
_COMPOSITE = frozenset((
    'UnaryExpression', 'BinaryExpression', 'LogicalExpression',
    'ConditionalExpression', 'ArrayExpression', 'ObjectExpression', 'Property',
    'TemplateLiteral', 'SequenceExpression', 'MemberExpression'))

# Anything here has effects we cannot see through.
_OPAQUE = frozenset((
    'CallExpression', 'NewExpression', 'TaggedTemplateExpression',
    'AwaitExpression', 'YieldExpression', 'ClassExpression', 'ClassDeclaration',
    'SpreadElement'))


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _flatten(seq):
    """Left-to-right list of non-sequence expressions in (possibly nested) seq."""
    out, stack = [], [seq]
    while stack:
        cur = stack.pop()
        if cur.kind == 'SequenceExpression':
            stack.extend(reversed(cur.children_in('expressions')))
        else:
            out.append(cur)
    return out


def _droppable(expr):
    return expr.kind in _DROPPABLE


def _is_reference_sensitive(expr):
    """True if `(0, expr)()` differs from `expr()` (this-binding / direct eval)."""
    if expr.kind in ('MemberExpression', 'ChainExpression'):
        return True
    return expr.kind == 'Identifier' and expr.attr('name') == 'eval'


def _read_children(parent):
    """Children that are *evaluated* as values (skips labels / plain keys)."""
    for _slot, child, role in parent.iter_children():
        if role == LABEL:
            continue
        if role == KEY and not parent.attr('computed'):
            continue
        yield child


def _negate(expr):
    if expr.kind == 'UnaryExpression' and expr.attr('operator') == '!':
        return clone(expr.get('argument'))
    return make('UnaryExpression', operator='!', prefix=True, argument=expr)


def _rebuild(n, replaced):
    """Fresh copy of `n` with the given {(slot, index): fresh_node} overrides."""
    fields = dict(n.attrs)
    for slot, arity, _role in slots_of(n.kind):
        value = n.children.get(slot)
        if value is None:
            continue
        if arity is MANY:
            fields[slot] = [
                replaced[(slot, i)] if (slot, i) in replaced
                else (clone(item) if item is not None else None)
                for i, item in enumerate(value)]
        else:
            fields[slot] = (replaced[(slot, None)] if (slot, None) in replaced
                            else clone(value))
    fresh = make(n.kind, **fields)
    fresh.origin = n.origin
    return fresh


def _target_names(target, out):
    if target is None:
        return
    k = target.kind
    if k == 'Identifier':
        out.add(target.attr('name'))
    elif k == 'ArrayPattern':
        for e in target.children_in('elements'):
            _target_names(e, out)
    elif k == 'ObjectPattern':
        for p in target.children_in('properties'):
            _target_names(p.get('value') if p.kind == 'Property' else p, out)
    elif k == 'RestElement':
        _target_names(target.get('argument'), out)
    elif k == 'AssignmentPattern':
        _target_names(target.get('left'), out)


def _mutable_names(root):
    """Names that are assigned/updated anywhere (name-based, scope-insensitive)."""
    names = set()
    for n in walk(root):
        k = n.kind
        if k == 'AssignmentExpression':
            _target_names(n.get('left'), names)
        elif k == 'UpdateExpression':
            _target_names(n.get('argument'), names)
        elif k in ('ForInStatement', 'ForOfStatement'):
            left = n.get('left')
            if left is not None and left.kind == 'VariableDeclaration':
                for d in left.children_in('declarations'):
                    _target_names(d.get('id'), names)
            else:
                _target_names(left, names)
    return names


def _candidate_hosts(root):
    """nids of host nodes that directly own at least one SequenceExpression."""
    hosts = set()
    for seq in walk(root, kinds={'SequenceExpression'}):
        cur = seq.parent
        while cur is not None:
            if cur.kind in _HOST_KINDS:
                hosts.add(cur.nid)
                break
            if cur.is_statement or cur.kind in FUNCTION_KINDS:
                break
            cur = cur.parent
    return hosts


class _Effects:
    __slots__ = ('writes', 'heap', 'opaque')

    def __init__(self):
        self.writes = set()   # identifier names assigned
        self.heap = False     # member writes / delete
        self.opaque = False   # calls etc.: may do anything

    @property
    def empty(self):
        return not (self.writes or self.heap or self.opaque)


def _effects(exprs):
    fx = _Effects()
    for root in exprs:
        if root.kind in FUNCTION_KINDS:
            continue  # creating a function has no effect
        for cur in walk(root, enter_functions=False):
            k = cur.kind
            if k == 'AssignmentExpression':
                left = cur.get('left')
                if left.kind == 'Identifier':
                    fx.writes.add(left.attr('name'))
                elif left.kind == 'MemberExpression':
                    fx.heap = True
                else:
                    fx.opaque = True  # destructuring: iterators / setters
            elif k == 'UpdateExpression':
                arg = cur.get('argument')
                if arg.kind == 'Identifier':
                    fx.writes.add(arg.attr('name'))
                else:
                    fx.heap = True
            elif k in _OPAQUE:
                fx.opaque = True
            elif k == 'UnaryExpression' and cur.attr('operator') == 'delete':
                fx.heap = True
    return fx


# ---------------------------------------------------------------------------
# The hoister
# ---------------------------------------------------------------------------

class _Hoister:
    def __init__(self, tree, aggressive=False, rewrite_loop_tests=True,
                 is_local_stable=None, max_rounds=64):
        self.tree = tree
        self.aggressive = aggressive
        self.rewrite_loop_tests = rewrite_loop_tests
        self.local_hook = is_local_stable or (lambda _n: False)
        self.max_rounds = max_rounds
        self.stats = Counter()
        self.skipped = 0
        self._mutable = set()

    # -- driver ------------------------------------------------------------

    def run(self):
        for _ in range(self.max_rounds):
            hosts = _candidate_hosts(self.tree.root)
            if not hosts:
                return True
            self._mutable = _mutable_names(self.tree.root)
            changed = False
            for n in list(walk(self.tree.root, kinds=_HOST_KINDS)):
                if not n.live or n.nid not in hosts:
                    continue
                try:
                    if self._process(n):
                        changed = True
                except RecursionError:
                    self.skipped += 1
            if not changed:
                return True
        return False

    def _process(self, n):
        k = n.kind
        if k in _SLOT_HOSTS:
            return self._hoist_slot(n, _SLOT_HOSTS[k])
        if k == 'ExpressionStatement':
            return self._expression_statement(n)
        if k in ('ReturnStatement', 'ThrowStatement'):
            return self._hoist_slot(n, 'argument')
        if k == 'WhileStatement':
            return self._loop_test(n)
        if k == 'ForStatement':
            init = n.get('init')
            if init is not None and init.kind != 'VariableDeclaration':
                if self._hoist_slot(n, 'init'):
                    return True
            return self._loop_test(n)
        if k == 'VariableDeclaration':
            return self._variable_declaration(n)
        if k == 'ExportDefaultDeclaration':
            decl = n.get('declaration')
            if decl is None or decl.kind in ('FunctionDeclaration', 'ClassDeclaration'):
                return False
            return self._hoist_slot(
                n, 'declaration',
                guard=lambda r: r.kind not in ('FunctionExpression', 'ClassExpression'))
        if k == 'ArrowFunctionExpression':
            return self._arrow(n)
        return False

    # -- statement hosts -----------------------------------------------------

    def _hoist_slot(self, stmt, slot, guard=None):
        expr = stmt.get(slot)
        if expr is None:
            return False
        r = self._lift(expr)
        if r is None:
            return False
        res, pre = r
        if guard is not None and not guard(res):
            return False
        self.tree.replace(expr, res)
        if pre:
            self._emit_before(stmt, self._expr_stmts(pre))
        self.stats[f'{stmt.kind}.{slot}'] += 1
        return True

    def _expression_statement(self, stmt):
        if stmt.attr('directive'):
            return False
        expr = stmt.get('expression')
        r = self._lift(expr)
        if r is not None:
            res, pre = r
            stmts = self._expr_stmts(pre)
            if not _droppable(res):
                stmts.append(make('ExpressionStatement', expression=res))
            self._replace_statement(stmt, stmts)
            self.stats['ExpressionStatement'] += 1
            return True
        return False

    def _loop_test(self, loop):
        test = loop.get('test')
        if test is None:
            return False
        r = self._lift(test)
        if r is None:
            return False
        res, pre = r
        if not pre:
            self.tree.replace(test, res)
            self.stats[f'{loop.kind}.test'] += 1
            return True
        if not self.rewrite_loop_tests:
            return False
        body = loop.get('body')
        guard = make('IfStatement', test=_negate(res),
                     consequent=make('BreakStatement', label=None), alternate=None)
        new_body = make('BlockStatement',
                        body=self._expr_stmts(pre) + [guard, clone(body)])
        self.tree.replace(body, new_body)
        if loop.kind == 'WhileStatement':
            self.tree.replace(test, make('Literal', value=True))
        else:
            self.tree.remove(test)
        self.stats[f'{loop.kind}.test'] += 1
        return True

    def _arrow(self, fn):
        body = fn.get('body')
        if body is None or body.kind == 'BlockStatement':
            return False
        r = self._lift(body)
        if r is None:
            return False
        res, pre = r
        if not pre:
            self.tree.replace(body, res)
        else:
            attrs = dict(fn.attrs)
            attrs['expression'] = False
            block = make('BlockStatement', body=self._expr_stmts(pre) + [
                make('ReturnStatement', argument=res)])
            new = make('ArrowFunctionExpression',
                       params=[clone(p) for p in fn.children_in('params')],
                       body=block, **attrs)
            new.origin = fn.origin
            self.tree.replace(fn, new)
        self.stats['ArrowFunctionExpression.body'] += 1
        return True

    # -- declarations ---------------------------------------------------------

    def _variable_declaration(self, decl):
        parent = decl.parent
        if parent is None:
            return False
        if parent.kind in ('ForInStatement', 'ForOfStatement') and decl.parent_slot == 'left':
            return False

        if parent.kind == 'ForStatement' and decl.parent_slot == 'init':
            out = self._lift_declaration(
                decl, allow_split=decl.attr('kind') == 'var')
            if not out:
                return False
            *before, new = out
            self.tree.replace(decl, new)
            if before:
                self._emit_before(parent, before)
            self.stats['VariableDeclaration (for-init)'] += 1
            return True

        if parent.kind == 'ExportNamedDeclaration':
            out = self._lift_declaration(decl, allow_split=False)
            if not out:
                return False
            *before, new = out
            self.tree.replace(decl, new)
            if before:
                self._emit_before(parent, before)
            self.stats['VariableDeclaration (export)'] += 1
            return True

        out = self._lift_declaration(decl, allow_split=True)
        if not out:
            return False
        self._replace_statement(decl, out)
        self.stats['VariableDeclarator.init'] += 1
        return True

    def _lift_declaration(self, decl, allow_split):
        """Statements replacing `decl`, or None. The last one is always a
        VariableDeclaration."""
        kind = decl.attr('kind')
        declarators = list(decl.children_in('declarations'))
        out, group, changed = [], [], False

        def flush():
            nonlocal group
            if group:
                out.append(make('VariableDeclaration', kind=kind, declarations=group))
                group = []

        for idx, d in enumerate(declarators):
            init, ident = d.get('init'), d.get('id')
            r = self._lift(init) if init is not None else None

            if r is None:
                group.append(clone(d))
                continue

            res, pre = r
            if pre:
                if group:
                    if not allow_split:
                        group.extend(clone(x) for x in declarators[idx:])
                        break
                    flush()
                out.extend(self._expr_stmts(pre))
            changed = True
            group.append(make('VariableDeclarator', id=clone(ident), init=res))

        if not changed:
            return None
        flush()
        return out

    # -- statement placement ---------------------------------------------------

    @staticmethod
    def _expr_stmts(exprs):
        return [make('ExpressionStatement', expression=e) for e in exprs]

    @staticmethod
    def _anchor(stmt):
        """The statement before which new statements may be inserted."""
        cur = stmt
        while cur.parent is not None:
            p = cur.parent
            if (p.kind in ('LabeledStatement', 'ExportNamedDeclaration',
                           'ExportDefaultDeclaration')
                    or (p.kind == 'ForStatement' and cur.parent_slot == 'init')
                    or (p.kind in ('ForInStatement', 'ForOfStatement')
                        and cur.parent_slot == 'left')):
                cur = p
            else:
                break
        return cur

    def _emit_before(self, stmt, statements):
        anchor = self._anchor(stmt)
        container = anchor.parent.children.get(anchor.parent_slot)
        if isinstance(container, list):
            self.tree.insert_before(anchor, statements)
        else:
            self.tree.replace(anchor, make(
                'BlockStatement', body=list(statements) + [clone(anchor)]))

    def _replace_statement(self, stmt, statements):
        container = stmt.parent.children.get(stmt.parent_slot)
        if isinstance(container, list):
            self.tree.replace_many(stmt, statements)
        elif not statements:
            self.tree.replace(stmt, make('EmptyStatement'))
        elif len(statements) == 1:
            self.tree.replace(stmt, statements[0])
        else:
            self.tree.replace(stmt, make('BlockStatement', body=list(statements)))

    # -- expression lifting ------------------------------------------------------
    #
    # _lift(n) -> None                      nothing to hoist inside n
    #          -> (new_n, pre)              fresh detached nodes; `pre` must run
    #                                       (in order) before `new_n` is evaluated

    def _lift(self, n):
        if n.kind == 'SequenceExpression':
            return self._lift_sequence(n)
        items = self._items(n)
        if not items:
            return None
        callee_key = {'CallExpression': ('callee', None),
                      'TaggedTemplateExpression': ('tag', None)}.get(n.kind)
        return self._lift_ordered(n, items, callee_key)

    def _lift_sequence(self, seq, this_sensitive=False):
        *head, last = _flatten(seq)
        pre = []
        for e in head:
            r = self._lift(e)
            if r is None:
                if not _droppable(e):
                    pre.append(clone(e))
            else:
                new_e, e_pre = r
                pre.extend(e_pre)
                if not _droppable(new_e):
                    pre.append(new_e)

        r = self._lift(last)
        sensitive = this_sensitive and _is_reference_sensitive(last)
        if sensitive and not pre and r is None:
            return None  # already the canonical `(0, o.m)` form
        if r is None:
            res, last_pre = clone(last), []
        else:
            res, last_pre = r
        pre.extend(last_pre)
        if sensitive:
            res = make('SequenceExpression',
                       expressions=[make('Literal', value=0), res])
        return res, pre

    def _lift_ordered(self, n, items, callee_key=None):
        replaced, pre, residuals = {}, [], []
        for key, child, check in items:
            if child is None:
                continue
            if key == callee_key and child.kind == 'SequenceExpression':
                r = self._lift_sequence(child, this_sensitive=True)
            else:
                r = self._lift(child)
            if r is None:
                residuals.append((check, child))
                continue
            new_child, child_pre = r
            if child_pre:
                fx = _effects(child_pre)
                if not all(chk(res, fx) for chk, res in residuals):
                    residuals.append((check, child))  # unsafe: keep it in place
                    continue
                pre.extend(child_pre)
            replaced[key] = new_child
            residuals.append((check, new_child))
        if not replaced:
            return None
        return _rebuild(n, replaced), pre

    def _items(self, n):
        """Unconditionally-evaluated children of `n`, in evaluation order, as
        (key, child, commute_check_for_the_residual)."""
        kind, C = n.kind, self._commutes

        def one(slot, child, chk=C):
            return ((slot, None), child, chk)

        if kind == 'UnaryExpression':
            arg, op = n.get('argument'), n.attr('operator')
            if arg is None:
                return []
            if arg.kind == 'SequenceExpression':
                if op == 'delete':
                    return []
                if op == 'typeof' and _flatten(arg)[-1].kind == 'Identifier':
                    return []
            return [one('argument', arg)]

        if kind == 'BinaryExpression':
            return [one('left', n.get('left')), one('right', n.get('right'))]
        if kind == 'LogicalExpression':
            return [one('left', n.get('left'))]
        if kind == 'ConditionalExpression':
            return [one('test', n.get('test'))]

        if kind == 'AssignmentExpression':
            left, right, op = n.get('left'), n.get('right'), n.attr('operator')
            if left.kind in ('ArrayPattern', 'ObjectPattern'):
                return [one('right', right)]       # RHS is evaluated first
            if op in ('&&=', '||=', '??='):
                return [one('left', left)]         # RHS is conditional
            compound = op != '='
            return [one('left', left,
                        lambda res, fx: self._target_commutes(res, fx, compound)),
                    one('right', right)]

        if kind in ('CallExpression', 'NewExpression'):
            if n.attr('optional'):
                return []
            items = [one('callee', n.get('callee'))]
            items += [(('arguments', i), a, C)
                      for i, a in enumerate(n.children_in('arguments'))]
            return items

        if kind == 'TaggedTemplateExpression':
            return [one('tag', n.get('tag')), one('quasi', n.get('quasi'))]

        if kind == 'MemberExpression':
            if n.attr('optional'):
                return []
            items = [one('object', n.get('object'))]
            if n.attr('computed'):
                items.append(one('property', n.get('property')))
            return items

        if kind == 'ArrayExpression':
            return [(('elements', i), e, C)
                    for i, e in enumerate(n.children_in('elements')) if e is not None]
        if kind == 'ObjectExpression':
            return [(('properties', i), p, C)
                    for i, p in enumerate(n.children_in('properties'))]
        if kind == 'Property':
            items = []
            if n.attr('computed'):
                items.append(one('key', n.get('key')))
            items.append(one('value', n.get('value')))
            return items
        if kind == 'TemplateLiteral':
            return [(('expressions', i), e, C)
                    for i, e in enumerate(n.children_in('expressions'))]
        if kind in ('SpreadElement', 'AwaitExpression', 'YieldExpression',
                    'UpdateExpression'):
            arg = n.get('argument')
            return [one('argument', arg)] if arg is not None else []
        return []  # ChainExpression, classes, functions, patterns, leaves

    # -- reordering safety -------------------------------------------------------

    def _commutes(self, p, fx):
        """Can the already-evaluated operand `p` be evaluated *after* code with
        effects `fx` instead of before it, without changing its meaning?"""
        if fx.empty:
            return True
        k = p.kind
        if k in _LEAF_SAFE:
            return True
        if k == 'Identifier':
            name = p.attr('name')
            if name in fx.writes:
                return False
            if fx.opaque:
                return (self.aggressive or name not in self._mutable
                        or self.local_hook(p))
            return True
        if k == 'MemberExpression' or (
                k == 'BinaryExpression' and p.attr('operator') in ('in', 'instanceof')):
            if p.attr('optional') or fx.heap or (fx.opaque and not self.aggressive):
                return False
        elif k == 'UnaryExpression':
            if p.attr('operator') == 'delete':
                return False
        elif k not in _COMPOSITE:
            return False
        return all(self._commutes(c, fx) for c in _read_children(p))

    def _target_commutes(self, left, fx, compound):
        if compound:
            return self._commutes(left, fx)
        if left.kind == 'MemberExpression':   # `o[k] = v`: o and k are read first
            return all(self._commutes(c, fx) for c in _read_children(left))
        return True


# ---------------------------------------------------------------------------
# Pass entry point
# ---------------------------------------------------------------------------

def lift_sequence_expressions(session: Session, **options) -> None:
    """
    Options (see module docstring): aggressive, rewrite_loop_tests,
    is_local_stable, max_rounds.
    Use functools.partial(lift_sequence_expressions, aggressive=True) to
    customise when passing to Session.run.
    """
    print("--- Applying Transformer: lift_sequence_expressions ---")
    hoister = _Hoister(session.tree, **options)
    converged = hoister.run()

    total = sum(hoister.stats.values())
    for label, count in sorted(hoister.stats.items()):
        print(f" |-> {label}: {count}")
    print(f" |-> Total sequence expressions rewritten: {total}")
    if hoister.skipped:
        print(f" |-> Skipped {hoister.skipped} statements (expression too deep)")
    if not converged:
        print(" |-> WARNING: hit max_rounds before reaching a fixed point")

    # Our edits replace large subtrees with clones; make sure incremental scope
    # tracking does not keep references to the old identifier nodes.
    if total and hasattr(session, '_on_tree_dirty'):
        session._on_tree_dirty()
