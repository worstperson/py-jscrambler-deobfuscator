"""
vmexpr_util -- shared flags, constants and helpers for the vmexpr passes.
"""

import re
import sys

from jsir import Session, Tree, node as make, clone, walk, values as V
from jsir.schema import FUNCTION_KINDS, TARGET, KEY, LABEL
from lifter.vmstructure import _negate_expr


ASSUME_PURE_GETS = True
ASSUME_TIMING_LOOPS_COMMUTE = True

# Variables accessed across closures that must not be eliminated as dead stores
CROSS_SCOPE_VARS = set()

BUILTINS = frozenset((
    'Date', 'Math', 'Object', 'Array', 'String', 'Number', 'Boolean', 'JSON', 'RegExp', 'Error',
    'Function', 'Symbol', 'Promise', 'Map', 'Set', 'parseInt', 'parseFloat', 'isNaN', 'isFinite',
    'encodeURIComponent', 'decodeURIComponent', 'escape', 'unescape', 'undefined', 'NaN', 'Infinity'))

# Note: 'eval' is excluded so globals_as_identifiers can turn vm_global.eval into eval
RESERVED = frozenset((
    'break case catch class const continue debugger default delete do else enum export extends false '
    'finally for function if import in instanceof new null return super switch this throw true try '
    'typeof var void while with yield let static await arguments implements interface package '
    'private protected public').split())

IDENT_RE = re.compile(r'^[A-Za-z_$][A-Za-z0-9_$]*$')
LOOP_KINDS = ('WhileStatement', 'DoWhileStatement', 'ForStatement', 'ForInStatement', 'ForOfStatement')


def ident(name):
    return make('Identifier', name=name)


def literal_node(value):
    """A literal for a CCP constant, or None when there is no literal syntax."""
    if value is V.UNDEF:
        return ident('undefined')
    if value is V.NULL or isinstance(value, (bool, str)):
        return make('Literal', value=value)
    if isinstance(value, float):
        if value != value or value in (float('inf'), float('-inf')):
            return None
        if value < 0:
            return make('UnaryExpression', operator='-', prefix=True, argument=make('Literal', value=-value))
        return make('Literal', value=value)
    return None


def _is_this_helper(fn):
    return fn.kind == 'ArrowFunctionExpression' and fn.get('body') is not None and fn.get('body').kind == 'ThisExpression'


def leaf_functions(root):
    out = []
    for fn in walk(root, kinds=FUNCTION_KINDS):
        body = fn.get('body')
        if body is None or body.kind != 'BlockStatement':
            continue
        inner_fns = [n for n in walk(body) if n is not fn and n.kind in FUNCTION_KINDS]
        if any(not _is_this_helper(n) for n in inner_fns):
            continue
        out.append(fn)
    return out


def tracked_bindings(index, fn):
    scope = index.scope_of(fn)
    if scope is None or scope.node is not fn or scope.has_dynamic_scope:
        return None
    return {name: b for name, b in scope.bindings.items()
            if b.kind == 'var' and b.promotable and name not in CROSS_SCOPE_VARS}


def scan(node, reads, writes):
    for _slot, child, role in node.iter_children():
        if child.kind == 'Identifier':
            if role is LABEL:
                continue
            if role is KEY and not node.attr('computed', False):
                continue
            if role is TARGET:
                if node.kind == 'AssignmentExpression':
                    writes.append(child)
                    if node.attr('operator', '=') != '=':
                        reads.append(child)
                elif node.kind == 'UpdateExpression':
                    reads.append(child)
                    writes.append(child)
                elif node.kind == 'VariableDeclarator' and node.get('init') is not None:
                    writes.append(child)
                continue
            reads.append(child)
        else:
            scan(child, reads, writes)


def names_read(node):
    reads, writes = [], []
    if node.kind == 'Identifier':
        return {node.attr('name')}
    scan(node, reads, writes)
    return {i.attr('name') for i in reads}


def names_written(node):
    reads, writes = [], []
    scan(node, reads, writes)
    return {i.attr('name') for i in writes}


def assignment_of(stmt):
    if stmt.kind != 'ExpressionStatement':
        return None
    e = stmt.get('expression')
    if e.kind == 'AssignmentExpression' and e.attr('operator', '=') == '=' \
            and e.get('left').kind == 'Identifier':
        return e.get('left').attr('name'), e
    return None


def pure(e):
    k = e.kind
    if k in ('Identifier', 'Literal', 'ThisExpression'):
        return True
    if k == 'MemberExpression':
        return ASSUME_PURE_GETS and pure(e.get('object')) and \
            (not e.attr('computed', False) or pure(e.get('property')))
    if k in ('BinaryExpression', 'LogicalExpression'):
        return pure(e.get('left')) and pure(e.get('right'))
    if k == 'UnaryExpression':
        return e.attr('operator') != 'delete' and pure(e.get('argument'))
    if k == 'ConditionalExpression':
        return all(pure(e.get(s)) for s in ('test', 'consequent', 'alternate'))
    if k == 'ArrayExpression':
        return all(el is None or (el.kind != 'SpreadElement' and pure(el)) for el in e.children_in('elements'))
    if k == 'SequenceExpression':
        return all(pure(x) for x in e.children_in('expressions'))
    return False


def builtin_named(e, name=None):
    if e.kind == 'Identifier':
        n = e.attr('name')
    elif e.kind == 'MemberExpression' and e.get('object').kind == 'Identifier' \
            and e.get('object').attr('name') == 'vm_global':
        p = e.get('property')
        n = p.attr('value') if e.attr('computed', False) and p.kind == 'Literal' else \
            (p.attr('name') if not e.attr('computed', False) else None)
    else:
        return False
    return n in BUILTINS if name is None else n == name


def declared_names(root):
    declaring = ('VariableDeclarator', 'FunctionDeclaration', 'FunctionExpression', 'ArrowFunctionExpression',
                 'ClassDeclaration', 'ClassExpression', 'CatchClause', 'RestElement', 'AssignmentPattern',
                 'ArrayPattern', 'ObjectPattern')
    out = set()
    for n in walk(root):
        if n.kind in declaring:
            for _slot, child, role in n.iter_children():
                if child.kind == 'Identifier' and role is TARGET:
                    out.add(child.attr('name'))
    return out


def heap_free(e, tracked):
    k = e.kind
    if k == 'Literal':
        return True
    if k == 'Identifier':
        return e.attr('name') in tracked or (e.attr('name') in BUILTINS)
    if k == 'MemberExpression':
        return builtin_named(e)
    if k in ('BinaryExpression', 'LogicalExpression'):
        return heap_free(e.get('left'), tracked) and heap_free(e.get('right'), tracked)
    if k == 'UnaryExpression':
        return e.attr('operator') != 'delete' and heap_free(e.get('argument'), tracked)
    return False


def body_lists(fn):
    out = [(fn.get('body'), 'body')]
    for b in walk(fn.get('body'), kinds={'BlockStatement'}):
        if b is not fn.get('body'):
            out.append((b, 'body'))
    return out


def used_names(fn):
    return {n.attr('name') for n in walk(fn, kinds={'Identifier'})}


def fresh(prefix, taken):
    i = 0
    while f'{prefix}{i}' in taken:
        i += 1
    taken.add(f'{prefix}{i}')
    return f'{prefix}{i}'


def declare(tree, fn, names):
    if not names:
        return
    decl = make('VariableDeclaration', kind='var', declarations=[
        make('VariableDeclarator', id=ident(n), init=None) for n in names])
    first = fn.get('body').children_in('body')
    if first:
        tree.insert_before(first[0], [decl])
    else:
        tree.set_slot(fn.get('body'), 'body', [decl])
