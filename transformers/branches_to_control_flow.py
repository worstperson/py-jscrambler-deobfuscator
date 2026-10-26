"""
branches_to_control_flow -- turn value-producing branches into statements.

    a && b();                 ->  if (a) { b(); }            (also ||, ??)
    c ? f() : g();            ->  if (c) { f(); } else { g(); }
    c ? f() : d ? g() : h();  ->  if (c) { f(); } else if (d) { g(); } else { h(); }

    x = c ? p : q;            ->  if (c) { x = p; } else { x = q; }
    x = a && b;               ->  x = a; if (x) { x = b; }
    var x = c ? p : q;        ->  var x; if (c) { x = p; } else { x = q; }
    return c ? p : q;         ->  if (c) { return p; } else { return q; }
    throw  c ? p : q;         ->  if (c) { throw p; } else { throw q; }

Two families, controlled separately because they differ in how much they change
readable code:

  statement_forms (default True)
      The expression's value is discarded (`a && b();`, `c ? f() : g();`).
      This is the classic minifier / obfuscator shape and the `if` is strictly
      clearer, so every such statement is converted.

  value_forms  ('sequence' | 'always' | 'never', default 'sequence')
      The value is consumed (assignment to an identifier, `var`/`let`
      initialiser, return, throw). `x = c ? a : b` is ordinary readable code, so
      by default these are only converted when a branch contains a
      SequenceExpression, i.e. when the result lets hoist_sequence_expressions
      do something it could not do before. 'always' converts all of them.

Convergence
    One run reaches a fixed point on its own:
      * statements are visited descendants-first (reverse of a pre-order walk),
        so converting an outer statement never throws away inner work when it
        clones the expression (e.g. a function body inside `a && (function(){...})()`);
      * statements created by a conversion are visited immediately;
      * sequences are flattened as they are met: in `c ? (a(), d ? x : y) : z`
        the arm becomes `a();` followed by the nested conditional, so no hoisting
        is needed in between. Heads of a comma list are always value-discarded
        and use the statement_forms rule; only the last element feeds the sink.
      * arrow functions with expression bodies are treated as a `return` sink.

    Recommended pipeline (the hoister can expose new `a && b();` statements
    when it splits a sequence nested inside a call, so it goes on both sides):

        session.run(hoist_sequence_expressions, branches_to_control_flow,
                    hoist_sequence_expressions)

Not converted: `const x = c ? ...` (needs its initialiser), member targets
(`o.p = c ? a : b`: the target is evaluated before the test), compound
assignment, `return a && b` / `return a || b` (would need a temporary),
declarations in `for` heads or under `export`.
"""

from collections import Counter

from jsir import Session, walk, node as make, clone


_KINDS = frozenset(('ExpressionStatement', 'ReturnStatement', 'ThrowStatement',
                    'VariableDeclaration', 'ArrowFunctionExpression'))

_DROPPABLE = frozenset((
    'Literal', 'ThisExpression', 'FunctionExpression', 'ArrowFunctionExpression'))


def _has_seq(expr):
    return next(walk(expr, kinds={'SequenceExpression'}, enter_functions=False),
                None) is not None


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


def _negate(expr):
    if expr.kind == 'UnaryExpression' and expr.attr('operator') == '!':
        return clone(expr.get('argument'))
    return make('UnaryExpression', operator='!', prefix=True, argument=expr)


def _truthiness_test(op, subject):
    """Condition under which the RHS of `subject <op> rhs` is evaluated."""
    if op == '&&':
        return subject
    if op == '||':
        return _negate(subject)
    # '??': evaluated when subject is null/undefined.
    # (Literal null is value=None; adjust if your bridge needs a `raw` attr.)
    return make('BinaryExpression', operator='==', left=subject,
                right=make('Literal', value=None))


def _block(statements):
    return make('BlockStatement', body=list(statements))


class _Expander:
    def __init__(self, tree, statement_forms=True, value_forms='sequence'):
        if value_forms not in ('sequence', 'always', 'never'):
            raise ValueError(f'value_forms must be sequence/always/never, got {value_forms!r}')
        self.tree = tree
        self.statement_forms = statement_forms
        self.value_forms = value_forms
        self.stats = Counter()
        self.skipped = 0

    # -- driver --------------------------------------------------------------

    def run(self):
        # Reverse pre-order == every node comes after all of its descendants.
        for n in reversed(list(walk(self.tree.root, kinds=_KINDS))):
            self._visit(n)

    def _visit(self, n):
        if not n.live:
            return
        try:
            created = self._process(n)
        except RecursionError:
            self.skipped += 1
            return
        for fresh in created:
            for m in reversed(list(walk(fresh, kinds=_KINDS))):
                self._visit(m)

    def _process(self, n):
        """Convert `n` if possible; returns the fresh nodes that replaced it."""
        k = n.kind
        if k == 'ExpressionStatement':
            if n.attr('directive'):
                return []
            return self._replace_with(n, self._expand(n.get('expression'), 'expr'))
        if k in ('ReturnStatement', 'ThrowStatement'):
            arg = n.get('argument')
            if arg is None:
                return []
            mode = 'return' if k == 'ReturnStatement' else 'throw'
            return self._replace_with(n, self._expand(arg, mode))
        if k == 'VariableDeclaration':
            return self._declaration(n)
        if k == 'ArrowFunctionExpression':
            return self._arrow(n)
        return []

    def _replace_with(self, stmt, statements):
        if not statements:
            return []
        self._replace_statement(stmt, statements)
        self.stats[stmt.kind] += 1
        return statements

    def _arrow(self, fn):
        body = fn.get('body')
        if body is None or body.kind == 'BlockStatement':
            return []
        stmts = self._expand(body, 'return')
        if not stmts:
            return []
        attrs = dict(fn.attrs)
        attrs['expression'] = False
        new = make('ArrowFunctionExpression',
                   params=[clone(p) for p in fn.children_in('params')],
                   body=_block(stmts), **attrs)
        new.origin = fn.origin
        self.tree.replace(fn, new)
        self.stats['ArrowFunctionExpression'] += 1
        return [new]

    # -- declarations ---------------------------------------------------------

    def _declaration(self, decl):
        parent = decl.parent
        if parent is None:
            return []
        if parent.kind in ('ExportNamedDeclaration', 'ForStatement',
                           'ForInStatement', 'ForOfStatement'):
            return []
        kind = decl.attr('kind')
        if kind not in ('var', 'let'):
            return []

        out, group, changed = [], [], False
        for d in decl.children_in('declarations'):
            init, ident = d.get('init'), d.get('id')
            stmts = None
            if init is not None and ident.kind == 'Identifier':
                stmts = self._expand(init, ('assign', ident))
            if not stmts:
                group.append(clone(d))
                continue
            group.append(make('VariableDeclarator', id=clone(ident), init=None))
            out.append(make('VariableDeclaration', kind=kind, declarations=group))
            group = []
            out.extend(stmts)
            changed = True
        if not changed:
            return []
        if group:
            out.append(make('VariableDeclaration', kind=kind, declarations=group))
        return self._replace_with(decl, out)

    # -- the conversion ---------------------------------------------------------

    def _want(self, expr, mode):
        if mode == 'expr':
            return self.statement_forms
        if self.value_forms == 'never':
            return False
        if self.value_forms == 'always':
            return True
        if expr.kind == 'ConditionalExpression':
            branches = (expr.get('consequent'), expr.get('alternate'))
        else:
            branches = (expr.get('right'),)
        return any(_has_seq(b) for b in branches)

    def _expand(self, expr, mode):
        """Statements equivalent to `expr` delivered to the sink `mode`
        ('expr' | 'return' | 'throw' | ('assign', identifier)), or None."""
        k = expr.kind

        if k == 'SequenceExpression':
            pieces, expanded = self._sequence(expr, mode)
            return pieces if expanded else None

        if (k == 'AssignmentExpression' and mode == 'expr'
                and expr.attr('operator') == '=' and expr.get('left').kind == 'Identifier'):
            return self._expand(expr.get('right'), ('assign', expr.get('left')))

        if k == 'ConditionalExpression' and self._want(expr, mode):
            cons = self._expand_or_sink(expr.get('consequent'), mode)
            alt = self._expand_or_sink(expr.get('alternate'), mode)
            alternate = alt[0] if (len(alt) == 1 and alt[0].kind == 'IfStatement') \
                else _block(alt)
            return [make('IfStatement', test=clone(expr.get('test')),
                         consequent=_block(cons), alternate=alternate)]

        if k == 'LogicalExpression' and mode not in ('return', 'throw') \
                and self._want(expr, mode):
            op, left = expr.attr('operator'), expr.get('left')
            body = _block(self._expand_or_sink(expr.get('right'), mode))
            if mode == 'expr':
                return [make('IfStatement', test=_truthiness_test(op, clone(left)),
                             consequent=body, alternate=None)]
            target = mode[1]
            return [self._sink(mode, clone(left)),
                    make('IfStatement', test=_truthiness_test(op, clone(target)),
                         consequent=body, alternate=None)]
        return None

    def _expand_or_sink(self, expr, mode):
        if expr.kind == 'SequenceExpression':
            return self._sequence(expr, mode)[0]   # always flatten inside an arm
        return self._expand(expr, mode) or [self._sink(mode, clone(expr))]

    def _sequence(self, seq, mode):
        """`a, b, c` delivered to `mode`: a and b are value-discarded statements,
        c goes to the sink. Returns (statements, whether anything was converted)."""
        *head, last = _flatten(seq)
        pieces, expanded = [], False
        for e in head:
            r = self._expand(e, 'expr')
            if r:
                pieces.extend(r)
                expanded = True
            elif e.kind not in _DROPPABLE:
                pieces.append(self._sink('expr', clone(e)))
        r = self._expand(last, mode)
        if r:
            pieces.extend(r)
            expanded = True
        elif not (mode == 'expr' and last.kind in _DROPPABLE):
            pieces.append(self._sink(mode, clone(last)))
        return pieces, expanded

    @staticmethod
    def _sink(mode, value):
        if mode == 'expr':
            return make('ExpressionStatement', expression=value)
        if mode == 'return':
            return make('ReturnStatement', argument=value)
        if mode == 'throw':
            return make('ThrowStatement', argument=value)
        return make('ExpressionStatement', expression=make(
            'AssignmentExpression', operator='=', left=clone(mode[1]), right=value))

    def _replace_statement(self, stmt, statements):
        container = stmt.parent.children.get(stmt.parent_slot)
        if isinstance(container, list):
            self.tree.replace_many(stmt, statements)
        elif not statements:
            self.tree.replace(stmt, make('EmptyStatement'))
        elif len(statements) == 1:
            self.tree.replace(stmt, statements[0])
        else:
            self.tree.replace(stmt, _block(statements))


def branches_to_control_flow(session: Session, **options) -> None:
    """
    Options: statement_forms (bool), value_forms ('sequence'|'always'|'never').
    Use functools.partial(branches_to_control_flow, value_forms='always') to
    customise when passing to Session.run.
    """
    print("--- Applying Transformer: branches_to_control_flow ---")
    expander = _Expander(session.tree, **options)
    expander.run()

    total = sum(expander.stats.values())
    for label, count in sorted(expander.stats.items()):
        print(f" |-> {label}: {count}")
    print(f" |-> Total branch expressions converted: {total}")
    if expander.skipped:
        print(f" |-> Skipped {expander.skipped} statements (expression too deep)")

    if total and hasattr(session, '_on_tree_dirty'):
        session._on_tree_dirty()
