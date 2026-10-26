"""
vmexpr_scaffold -- VM scaffolding: vm_locals, vm_global, scope plumbing, factories and child calls.
"""

import re
import sys

from jsir import Session, Tree, node as make, clone, walk, values as V
from jsir.schema import FUNCTION_KINDS, TARGET, KEY, LABEL
from lifter.vmstructure import _negate_expr
from lifter.transformers import vmexpr_util
from lifter.transformers.vmexpr_util import (
    IDENT_RE,
    RESERVED,
    declare,
    declared_names,
    ident,
    leaf_functions,
    used_names,
)


def _is_scope_target(node):
    curr = node
    while curr is not None and curr.kind == 'MemberExpression':
        curr = curr.get('object')
    return curr is not None and curr.kind == 'Identifier' and curr.attr('name') in ('vm_scope', 'vm_scope0')


def _indexes_scope(root):
    for m in walk(root, kinds={'MemberExpression'}):
        if _is_scope_target(m):
            return True
    return False


_HAD_INDEXED_SCOPE = False

# Wrap vm_main as a self-executing function expression (IIFE) when implied nesting is detected
IIFE_MAIN = True


def _build_scope_hierarchy(root, factories):
    """Reconstruct the parent-child closure hierarchy implied by factory calls."""
    parents = {}
    for fname, (_fdecl, inner) in factories.items():
        for call in walk(root, kinds={'CallExpression'}):
            c = call.get('callee')
            if c.kind == 'Identifier' and c.attr('name') == fname:
                curr = call.parent
                while curr is not None:
                    if curr.kind in FUNCTION_KINDS and any(curr is f[1] for f in factories.values()):
                        parents[inner] = curr
                        break
                    curr = curr.parent

    ancestors = {}
    for _fname, (_fdecl, inner) in factories.items():
        chain = [inner]
        while chain[-1] in parents:
            chain.append(parents[chain[-1]])
        ancestors[inner] = list(reversed(chain))
    return ancestors, parents


def promote_locals(session):
    global _HAD_INDEXED_SCOPE
    tree = session.tree
    root = tree.root
    factories = _factories(root)

    if not _indexes_scope(root):
        # Fast path for flat, independent functions (sample.structure)
        for fn in leaf_functions(root):
            sites, other = [], 0
            for n in walk(fn.get('body'), kinds={'Identifier'}):
                if n.attr('name') != 'vm_locals':
                    continue
                p = n.parent
                if p.kind == 'MemberExpression' and p.get('object') is n and p.attr('computed', False) \
                        and p.get('property').kind == 'Literal' and isinstance(p.get('property').attr('value'), float) \
                        and p.get('property').attr('value') == int(p.get('property').attr('value')) >= 0:
                    sites.append(p)
                elif p.kind == 'VariableDeclarator' and p.get('id') is n:
                    pass
                elif p.kind == 'ArrayExpression' and p.parent is not None and p.parent.kind == 'VariableDeclarator' \
                        and p.parent.get('id').attr('name') == 'vm_scope':
                    pass
                else:
                    other += 1
            if other or not sites:
                continue
            taken = used_names(fn)
            names = {}
            for m in sites:
                k = int(m.get('property').attr('value'))
                if k not in names:
                    name = f'l{k}'
                    while name in taken and name not in names.values():
                        name += '_'
                    taken.add(name)
                    names[k] = name
            for m in sites:
                tree.replace(m, ident(names[int(m.get('property').attr('value'))]))
            declare(tree, fn, [names[k] for k in sorted(names)])
        return

    # Scope-indexed path (sample2 with implied nesting)
    _HAD_INDEXED_SCOPE = True
    ancestors, _parents = _build_scope_hierarchy(root, factories)
    inner_fns = [f[1] for f in factories.values()]

    # 1. Strip synthetic scope operations in catch blocks (e.g. vm_scope[3][0] = ...)
    for s in [s for s in walk(root, kinds={'ExpressionStatement'})]:
        e = s.get('expression')
        if e.kind == 'AssignmentExpression' and _is_scope_target(e.get('left')):
            tree.remove(s)

    # 2. Assign unique local variable names per function to avoid sibling collisions
    var_map = {}
    for fn in inner_fns:
        depth = len(ancestors.get(fn, [fn])) - 1
        fn_name = fn.get('id').attr('name') if fn.get('id') is not None else f'fn{depth}'
        prefix = f'l_{fn_name}_' if depth > 0 else 'l'
        fn_sites = []
        for n in walk(fn.get('body'), kinds={'Identifier'}):
            if n.attr('name') != 'vm_locals':
                continue
            p = n.parent
            if p.kind == 'MemberExpression' and p.get('object') is n and p.attr('computed', False) \
                    and p.get('property').kind == 'Literal' and isinstance(p.get('property').attr('value'), float):
                k = int(p.get('property').attr('value'))
                fn_sites.append((p, k))
                if (id(fn), k) not in var_map:
                    var_map[(id(fn), k)] = f'{prefix}{k}'
        for p, k in fn_sites:
            tree.replace(p, ident(var_map[(id(fn), k)]))
        fn_names = {var_map[(id(fn), k)] for _p, k in fn_sites}
        declare(tree, fn, sorted(fn_names))

    # 3. Rewrite `vm_scope[row][col]` to the target ancestor's promoted variable
    for fn in inner_fns:
        chain = ancestors.get(fn, [fn])
        for m in [m for m in walk(fn.get('body'), kinds={'MemberExpression'})]:
            if not m.live:
                continue
            o = m.get('object')
            if o.kind == 'MemberExpression' and o.get('object').kind == 'Identifier' \
                    and o.get('object').attr('name') == 'vm_scope' \
                    and o.get('property').kind == 'Literal' and isinstance(o.get('property').attr('value'), float):
                row = int(o.get('property').attr('value'))
                if m.get('property').kind == 'Literal' and isinstance(m.get('property').attr('value'), float):
                    col = int(m.get('property').attr('value'))
                    if row < len(chain):
                        target_fn = chain[row]
                        target_var = var_map.get((id(target_fn), col))
                        if target_var:
                            vmexpr_util.CROSS_SCOPE_VARS.add(target_var)
                            tree.replace(m, ident(target_var))


def globals_as_identifiers(session):
    tree = session.tree
    declared = declared_names(tree.root)
    for fn in leaf_functions(tree.root):
        for m in [m for m in walk(fn.get('body'), kinds={'MemberExpression'})]:
            o = m.get('object')
            if not (m.live and o.kind == 'Identifier' and o.attr('name') == 'vm_global'):
                continue
            key = None
            if m.attr('computed', False) and m.get('property').kind == 'Literal':
                key = m.get('property').attr('value')
            elif not m.attr('computed', False):
                key = m.get('property').attr('name')
            if isinstance(key, str) and IDENT_RE.match(key) and key not in declared and key not in RESERVED:
                tree.replace(m, ident(key))


INLINE_CHILDREN = False
_SYNTHETIC = re.compile(r'(^fn_|_c\d+(_\d+)*$)')


def _factories(program):
    out = {}
    if program.kind != 'Program':
        return out
    for d in program.children_in('body'):
        if d.kind != 'FunctionDeclaration' or len(d.children_in('params')) != 1:
            continue
        stmts = d.get('body').children_in('body')
        if len(stmts) == 1 and stmts[0].kind == 'ReturnStatement' and stmts[0].get('argument') is not None \
                and stmts[0].get('argument').kind == 'FunctionExpression' and stmts[0].get('argument').get('id') is not None:
            out[d.get('id').attr('name')] = (d, stmts[0].get('argument'))
    return out


def _scope_arg(arg):
    if arg.kind == 'Identifier' and arg.attr('name') == 'vm_scope':
        return arg
    els = arg.children_in('elements') if arg.kind == 'ArrayExpression' else []
    if len(els) == 1 and els[0].kind == 'SpreadElement' and els[0].get('argument').kind == 'Identifier' \
            and els[0].get('argument').attr('name') == 'vm_scope':
        return els[0].get('argument')
    return None


def _scaffolding_is_inert(root, factories):
    if _indexes_scope(root):
        return False
    child_args = set()
    for call in walk(root, kinds={'CallExpression'}):
        c = call.get('callee')
        args = call.children_in('arguments')
        if c.kind == 'Identifier' and c.attr('name') in factories and len(args) == 1 and _scope_arg(args[0]) is not None:
            child_args.add(id(_scope_arg(args[0])))
    for n in walk(root, kinds={'Identifier'}):
        name = n.attr('name')
        p = n.parent
        if name == 'vm_scope':
            if not ((p.kind == 'VariableDeclarator' and p.get('id') is n) or id(n) in child_args):
                return False
        elif name == 'vm_locals':
            ok = p.kind == 'VariableDeclarator' and p.get('id') is n
            ok = ok or (p.kind == 'ArrayExpression' and p.parent.kind == 'VariableDeclarator'
                        and p.parent.get('id').attr('name') == 'vm_scope')
            if not ok:
                return False
        elif name == 'vm_scope0':
            ok = p in tuple(f[0] for f in factories.values()) or p.kind == 'FunctionDeclaration'
            ok = ok or (p.kind == 'SpreadElement' and p.parent.kind == 'ArrayExpression'
                        and p.parent.parent.kind == 'VariableDeclarator'
                        and p.parent.parent.get('id').attr('name') == 'vm_scope')
            if not ok:
                return False
    return True


def unwrap_vm_scaffolding(session, inline_children=None, iife_main=None):
    global _HAD_INDEXED_SCOPE
    if inline_children is None:
        inline_children = INLINE_CHILDREN
    if iife_main is None:
        iife_main = IIFE_MAIN and _HAD_INDEXED_SCOPE
    tree = session.tree
    root = tree.root
    factories = _factories(root)
    names = [f[1].get('id').attr('name') for f in factories.values()]
    if not factories or len(set(names)) != len(names) or not _scaffolding_is_inert(root, factories):
        return

    _ancestors, parents = _build_scope_hierarchy(root, factories) if _HAD_INDEXED_SCOPE else ({}, {})

    # 1. Child factory calls become references to the inner function
    for call in [c for c in walk(root, kinds={'CallExpression'})]:
        c = call.get('callee')
        args = call.children_in('arguments')
        if call.live and c.kind == 'Identifier' and c.attr('name') in factories and len(args) == 1 \
                and _scope_arg(args[0]) is not None:
            tree.replace(call, ident(factories[c.attr('name')][1].get('id').attr('name')))

    # 2. Clean declarations of VM scaffolding
    index = session.scopes
    for fn in [n for n in walk(root, kinds=FUNCTION_KINDS)]:
        scope = index.scope_of(fn)
        body = fn.get('body')
        if scope is None or body is None or body.kind != 'BlockStatement':
            continue
        for decl in [d for d in body.children_in('body') if d.kind == 'VariableDeclaration']:
            keep = []
            for d in decl.children_in('declarations'):
                name = d.get('id').attr('name')
                b = scope.bindings.get(name)
                inert = name in ('vm_locals', 'vm_scope', 'vm_this_fn') or (
                    name == 'vm_global' and b is not None and not b.reads)
                if inert and (name in ('vm_global', 'vm_this_fn') or b is None or not [
                        r for r in b.reads if not (r.parent.kind in ('SpreadElement', 'ArrayExpression'))]):
                    continue
                keep.append(d)
            if len(keep) != len(decl.children_in('declarations')):
                if keep:
                    tree.set_slot(decl, 'declarations', keep)
                else:
                    tree.remove(decl)

    # 3. Replace factories (nesting child functions inside parents when implied nesting was detected)
    for name, (decl, inner) in list(factories.items()):
        if _HAD_INDEXED_SCOPE and inner in parents:
            new = make('FunctionDeclaration', id=ident(inner.get('id').attr('name')),
                       params=[clone(p) for p in inner.children_in('params')], body=clone(inner.get('body')),
                       generator=inner.attr('generator', False), isAsync=inner.attr('isAsync', False))
            parent_fn = parents[inner]
            first_body = parent_fn.get('body').children_in('body')
            if first_body:
                tree.insert_before(first_body[0], [new])
            else:
                tree.set_slot(parent_fn.get('body'), 'body', [new])
            tree.remove(decl)
        elif iife_main and name == 'vm_main':
            # Self-executing root: wrap as an IIFE (function () { ... })()
            iife_expr = make('FunctionExpression', id=None, params=[], body=clone(inner.get('body')),
                             generator=inner.attr('generator', False), isAsync=inner.attr('isAsync', False))
            call_stmt = make('ExpressionStatement', expression=make('CallExpression', callee=iife_expr, arguments=[]))
            tree.replace(decl, call_stmt)
        else:
            new = make('FunctionDeclaration', id=ident(inner.get('id').attr('name')),
                       params=[clone(p) for p in inner.children_in('params')], body=clone(inner.get('body')),
                       generator=inner.attr('generator', False), isAsync=inner.attr('isAsync', False))
            tree.replace(decl, new)

    # 4. Single-use children, written out where they are used
    while inline_children:
        index = session.scopes
        pscope = index.scope_of(tree.root)
        target = None
        for d in tree.root.children_in('body'):
            if d.kind != 'FunctionDeclaration' or not _SYNTHETIC.search(d.get('id').attr('name')):
                continue
            b = pscope.bindings.get(d.get('id').attr('name')) if pscope is not None else None
            if b is not None and len(b.reads) == 1 and not any(n is b.reads[0] for n in walk(d)):
                target = (d, b.reads[0])
                break
        if target is None:
            break
        d, ref = target
        expr = make('FunctionExpression', id=None, params=[clone(p) for p in d.children_in('params')],
                    body=clone(d.get('body')), generator=d.attr('generator', False), isAsync=d.attr('isAsync', False))
        tree.replace(ref, expr)
        tree.remove(d)
