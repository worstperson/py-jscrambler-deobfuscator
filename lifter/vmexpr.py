"""
vmexpr -- expression recovery for structured lifter output.

vmstructure gives statements the right shape; what is left reads like a stack
machine: every operation bound to a numbered temporary, stack slots reused for
unrelated values, and the VM's register file spelled `vm_locals[k]`.

    v37 = vm_global["m2pNw"];  vm_locals[7] = v37;  v38 = v37["e6"];
    v39 = vm_locals[7];        v40 = v38.call(v39);

This module turns that back into expressions, as a list of jsir passes
(`(session) -> None`, the shape Session.run expects) driven to a fixpoint:

  recover_arguments  array copy of the rest parameter + prototype splice -> `s = [arguments]`
  sink_joins         a store after an if/else that every arm feeds moves into the arms
  loop_tail_guards   `if (c) continue; rest` at a loop tail -> `if (!c) { rest }`
  promote_locals     vm_locals[k] -> an ordinary variable, when nothing can index
                     the locals array behind our back (no vm_scope[...] anywhere)
  merge_selects      `if (c) x = a; else x = b;`  ->  `x = c ? a : b`
                     `x = a; if (a) x = b;`       ->  `x = a && b`
  split_webs         a reused slot becomes one fresh variable per def->use pair
                     (liveness-based), so each is single-write and can be inlined
  propagate_copies   `x = y` ... reads of x -> y, straight-line and across branches
  fold_constants     jsir's CCP: constant reads become literals, constant
                     branches collapse
  recover_calls      `o[k].call(o, a)` -> `o[k](a)`, `o["id"]` -> `o.id`
  remove_dead_stores `x = f()` with x never read -> `f()`; pure ones vanish
  inline_temps       single-use temporaries folded into their one use
  rematerialize_reads   `state = o[3]` read at several uses -> `o[3]` at each, when o is a
                     local nothing else reaches and no write to o[3] can run between
  compound_assignments  x = x + y -> x += y
  flatten_else_if / loop_conditions / prune_declarations   tidying

Soundness
---------
inline_temps is the pass that can go wrong, because it moves evaluation. It only
ever moves a def `v = E` forward into the statement that reads v, and only when

  * v is written once and read once (Binding.writes / Binding.reads);
  * everything between the def and the use is itself a pending def that is being
    moved the same way, so the group stays in its original order. The use must
    consume a contiguous *suffix* of the pending run, in evaluation order;
  * nothing the use evaluates *before* the moved expression can observe the move:
    reads of local variables are fine (a call cannot write a non-escaping local),
    property reads are fine only when the moved expression is itself pure, and
    anything with an effect blocks the move;
  * the read is not under a conditional operand (`?:`, right side of `&&`).

Two assumptions are on by default, as in most deobfuscation work:
property reads and arithmetic have no side effects (no getters, no valueOf).
Set `ASSUME_PURE_GETS = False` to keep every read where it was.

`vm_global["x"]` -> `x` is a separate, opt-in pass (globals_as_identifiers): it
changes a missing global from `undefined` into a ReferenceError.

Usage
    python vmexpr.py STRUCTURED.js [-o OUT.js] [--keep-globals] [--keep-scaffolding] [--inline-children]
    from vmexpr import simplify;  simplify(session)
"""

import re
import sys

from jsir import Session, Tree, node as make, clone, walk, values as V
from jsir.schema import FUNCTION_KINDS, TARGET, KEY, LABEL
from lifter.vmstructure import _negate_expr

from lifter.transformers import vmexpr_util
from lifter.transformers.vmexpr_util import (
    ASSUME_PURE_GETS,
    ASSUME_TIMING_LOOPS_COMMUTE,
    BUILTINS,
    RESERVED,
    IDENT_RE,
    LOOP_KINDS,
    ident,
    literal_node,
    leaf_functions,
    tracked_bindings,
    scan,
    names_read,
    names_written,
    assignment_of,
    pure,
    builtin_named,
    declared_names,
    heap_free,
    body_lists,
    used_names,
    fresh,
    declare,
)
from lifter.transformers.vmexpr_liveness import (
    _Liveness,
    _escapes,
)
from lifter.transformers.vmexpr_dataflow import (
    split_webs,
    propagate_copies,
    _Copies,
    fold_constants,
    remove_dead_stores,
    prune_declarations,
    fold_declarations,
)
from lifter.transformers.vmexpr_inline import (
    _events,
    _Inliner,
    inline_temps,
    _key_of,
    _stores_to,
    _clean_path,
    rematerialize_reads,
)
from lifter.transformers.vmexpr_idioms import (
    _single_assignment,
    merge_selects,
    recover_calls,
    _COMPOUND,
    _simple_ref,
    _same_ref,
    compound_assignments,
    _member_name,
    _call_of,
    _empty,
    _get_prototype_of,
    recover_arguments,
    _if_leaves,
    sink_joins,
)
from lifter.transformers.vmexpr_control import (
    _is_continue,
    _tail_block,
    _tail_if,
    loop_tail_guards,
    flatten_else_if,
    _ends_in_jump,
    tidy_conditions,
    _normal_test,
    _FLIP,
    loop_conditions,
)
from lifter.transformers.vmexpr_scaffold import (
    _indexes_scope,
    promote_locals,
    globals_as_identifiers,
    _SYNTHETIC,
    _factories,
    _scope_arg,
    _scaffolding_is_inert,
    unwrap_vm_scaffolding,
)


SETUP = (promote_locals, globals_as_identifiers)
CORE = (merge_selects, sink_joins, split_webs, propagate_copies, fold_constants, recover_calls,
        remove_dead_stores, recover_arguments, inline_temps, rematerialize_reads, compound_assignments,
        tidy_conditions, loop_tail_guards, loop_conditions, flatten_else_if, prune_declarations)
FINAL = (unwrap_vm_scaffolding,)
LAST = (prune_declarations, fold_declarations)


def simplify(session, globals=True, unwrap=True, inline_children=True, limit=12):
    """Run the whole pipeline to a fixpoint. Returns the session.

    globals: vm_global.x -> x          unwrap: remove the VM scaffolding (factories, vm_locals,
    vm_scope, vm_global, child factory calls)"""
    vmexpr_util.CROSS_SCOPE_VARS.clear()
    session.run(promote_locals)
    if globals:
        session.run(globals_as_identifiers)
    session.run_until_stable(*CORE, limit=limit)
    if unwrap:
        from lifter.transformers import vmexpr_scaffold
        vmexpr_scaffold.INLINE_CHILDREN = bool(inline_children)
        session.run(*FINAL)
    session.run(*LAST)
    return session


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    from lifter import vmlift
    session = Session.parse(open(argv[1]).read())
    simplify(session, globals='--keep-globals' not in argv, unwrap='--keep-scaffolding' not in argv,
             inline_children='--inline-children' in argv)
    text = vmlift.to_js(session.tree.root)
    if '-o' in argv:
        open(argv[argv.index('-o') + 1], 'w').write(text + '\n')
    else:
        print(text)
    print(session.report(), file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

