"""
extract.py -- pull one node out of the tree together with everything it needs
to run standalone, as a fresh, self-contained Program.

This exists to hand isolated snippets to a real interpreter (js2py) for
concrete execution: the thing CCP's abstract lattice can't do once a function
does anything past its model (loops over concrete data, bitwise hashing,
table lookups, ...). It only builds the standalone program; it does not run
anything itself.

Two kinds of dependency edge are resolved, because this codebase has two
kinds of binding:

  * Ordinary scope-bound free variables (session.scopes) -- walk up to the
    declaration. A function-valued binding is copied wholesale and its own
    free variables recursed into. A plain variable prefers a CCP concrete
    value (session.constant_of) over copying the real declaration, since
    that's usually the only thing available for a param or loop variable;
    falls back to copying the real declaration only when it's safely at
    Program scope (nothing else is needed to make sense of it in isolation).

  * `_global[key]` accesses, which don't resolve through session.scopes at
    all -- `_global` is a deliberately free name introduced by prepare.py.
    These resolve through session.factory_entries instead, pulling in the
    whole `_global[key] = (function(){...})();` IIFE assignment that
    produced that key's object.

Anything neither strategy can resolve is reported in `unresolved` rather than
silently producing a snippet that will ReferenceError at eval time.
"""

from jsir import Session, walk, clone, to_esprima
from jsir.schema import slots_of, KEY, LABEL, TARGET
from jsir.tree import Tree, node as make

from transformers.global_factory_functions import get_member_key
from transformers.fold_constants import _value_to_node


class ExtractionResult:
    def __init__(self, program, unresolved):
        self.program = program        # a standalone jsir Program node
        self.unresolved = unresolved  # names/keys we couldn't supply; caller
                                       # must define these before eval'ing, or
                                       # treat the extraction as unusable

    def source(self, generate):
        """generate: a callable(esprima_ast, **opts) -> str, e.g.
        escodegen_python.escodegen.generate -- kept as a parameter rather
        than importing escodegen directly, matching Session.generate()."""
        return generate(to_esprima(Tree(self.program)))


def _identifier_role(identifier):
    parent = identifier.parent
    if parent is None:
        return None
    for slot, _arity, role in slots_of(parent.kind):
        if slot == identifier.parent_slot:
            return role
    return None


def _is_value_reference(identifier):
    """True if this Identifier occurrence is actually evaluated (a read), as
    opposed to a property key, a label, or a bare declaration target -- none
    of which name a dependency that needs supplying."""
    role = _identifier_role(identifier)
    if role is LABEL:
        return False
    if role is KEY:
        return bool(identifier.parent.attr('computed', False))
    if role is TARGET:
        return False
    return True


def _is_within(node, ancestor):
    current = node
    while current is not None:
        if current is ancestor:
            return True
        current = current.parent
    return False


def _enclosing_declaration_statement(declarator):
    parent = declarator.parent
    return parent if parent is not None and parent.kind == 'VariableDeclaration' else declarator


def _as_statement(expr_node):
    if expr_node.is_statement:
        return expr_node
    parent = expr_node.parent
    if parent is not None and parent.kind == 'ExpressionStatement' and parent.get('expression') is expr_node:
        return parent
    # Synthesizing a wrapper: clone first, never adopt a live node into a new
    # parent, or we'd silently detach it from wherever it actually lives.
    return make('ExpressionStatement', expression=clone(expr_node))


def _make_value_declaration(name, value):
    literal_node = _value_to_node(value)
    if literal_node is None:
        return None
    declarator = make('VariableDeclarator', id=make('Identifier', name=name), init=literal_node)
    return make('VariableDeclaration', kind='var', declarations=[declarator])


def _resolve_binding(session, binding, add_statement, unresolved):
    decl = binding.declaration

    # A real function: always prefer the actual body over any value CCP
    # might report, since js2py needs to actually execute it.
    if decl is not None:
        if decl.kind == 'FunctionDeclaration':
            add_statement('function', decl)
            return
        if decl.kind == 'VariableDeclarator':
            init = decl.get('init')
            if init is not None and init.kind in ('FunctionExpression', 'ArrowFunctionExpression'):
                add_statement('function', _enclosing_declaration_statement(decl))
                return
        if decl.kind == 'ClassDeclaration':
            add_statement('class', decl)
            return

    # A concrete value from any read -- covers params, loop variables, catch
    # bindings, plain vars: anything CCP could pin down regardless of where
    # it's actually declared.
    value = None
    for read in binding.reads:
        value = session.constant_of(read)
        if value is not None:
            break
    if value is not None:
        decl_stmt = _make_value_declaration(binding.name, value)
        if decl_stmt is not None:
            add_statement('value', decl_stmt)
            return

    # A plain var/let/const at Program scope: safe to copy wholesale even
    # without a known value, since nothing outside our control defines it.
    if decl is not None and decl.kind == 'VariableDeclarator' and binding.scope.node.kind == 'Program':
        add_statement('var', _enclosing_declaration_statement(decl))
        return

    unresolved.add(binding.name)


def extract(session: Session, target) -> ExtractionResult:
    runtime_object = getattr(session, 'runtime_object', None)
    factory_by_key = {
        entry['global_key']: entry
        for entry in (getattr(session, 'factory_entries', None) or [])
    }

    collected = []          # [(kind, node)] in discovery order
    seen_stmt_nids = set()
    seen_bindings = set()
    seen_factory_keys = set()
    unresolved = set()
    included_roots = [target]
    queue = [target]

    def add_statement(kind, stmt_node):
        if stmt_node is None or stmt_node.nid in seen_stmt_nids:
            return
        seen_stmt_nids.add(stmt_node.nid)
        collected.append((kind, stmt_node))
        included_roots.append(stmt_node)
        queue.append(stmt_node)

    while queue:
        node = queue.pop()
        for identifier in walk(node, kinds={'Identifier'}):
            if not _is_value_reference(identifier):
                continue

            binding = session.scopes.binding_of(identifier)

            if binding is not None:
                if binding.declaration is not None and \
                        any(_is_within(binding.declaration, root) for root in included_roots):
                    continue
                if binding in seen_bindings:
                    continue
                seen_bindings.add(binding)
                _resolve_binding(session, binding, add_statement, unresolved)
                continue

            name = identifier.attr('name')

            if runtime_object is not None and name == runtime_object:
                parent = identifier.parent
                if parent is not None and parent.kind == 'MemberExpression' and parent.get('object') is identifier:
                    key = get_member_key(parent)
                    entry = factory_by_key.get(key)
                    if entry is not None:
                        if key not in seen_factory_keys:
                            seen_factory_keys.add(key)
                            add_statement('factory', _as_statement(entry['assign_node']))
                        continue
                unresolved.add(runtime_object)
                continue

            # A genuinely free name: could be a real JS global (Math, String,
            # ...) or something obfuscation-internal we never found a
            # binding for. Can't tell the difference here.
            unresolved.add(name)

    functions = [clone(n) for kind, n in collected if kind in ('function', 'class', 'factory')]
    values = [clone(n) for kind, n in collected if kind not in ('function', 'class', 'factory')]

    body = functions + values
    if target.is_statement:
        body.append(clone(target))
    else:
        body.append(make('VariableDeclaration', kind='var', declarations=[
            make('VariableDeclarator', id=make('Identifier', name='__jsir_result__'), init=clone(target)),
        ]))

    program = make('Program', body=body, sourceType='script')
    return ExtractionResult(program, unresolved)
