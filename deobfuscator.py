import time
from pathlib import Path

from jsir import Session

from transformers.unflatten_control_flow import unflatten_control_flow
from transformers.eval_unpack import unpack_eval_packer
from transformers.runtime_global_getter import runtime_global_getter
from transformers.find_runtime_functions import find_runtime_functions
from transformers.inline_wrapper_functions import inline_wrapper_functions, resolve_wrapper_aliases
from transformers.remove_code_locks import environmental_scorer, remove_code_locks
from transformers.cyclical_array_generators import find_cyclical_array_generators
from transformers.fold_boolean_comparisons import fold_boolean_comparisons
from transformers.restore_local_variables import restore_local_variables
from transformers.solve_state_machine import solve_state_machine
from transformers.fold_constants import fold_constants, dead_code_elimination
from transformers.inline_alias_lookups import find_alias_lookups, inline_alias_lookups
from transformers.solve_string_array import solve_string_array
from transformers.simplify_boolean_logic import simplify_boolean_logic
from transformers.resolve_hash_guards import resolve_hash_guards
from transformers.remove_dead_code_after_terminator import remove_dead_code_after_terminator
from transformers.restore_switch_statements import restore_switch_statements
from transformers.lift_sequence_expressions import lift_sequence_expressions
from transformers.branches_to_control_flow import branches_to_control_flow
from transformers.unnest_assignment_expressions import unnest_assignment_expressions
from transformers.restore_for_loops import restore_for_loops
from transformers.lift_vm import lift_vm
from transformers.inline_call_wrappers import inline_call_wrappers
from transformers.remove_runtime_submodules import remove_runtime_submodules
from transformers.fold_arrays import fold_arrays
from transformers.propagate_aliases import propagate_aliases
from transformers.propagate_runtime_globals import propagate_runtime_globals
from transformers.rename_vars import rename_vars
from transformers.propagate_runtime_aliases import propagate_runtime_aliases
from transformers.inline_trap_functions import inline_trap_functions
from transformers.remove_eval_traps import remove_eval_traps

import patches.escodegen
import patches.esprima


input_dir = Path("samples")
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

for file_path in input_dir.glob("*.js"):
    print(file_path)

    start_time = time.perf_counter()

    file_contents = file_path.read_text(encoding="utf-8")

    session = Session.parse(file_contents)
    session.string_functions = []
    session.dce_keep = []
    session.oracle.runtime_object = None

    session.run(unpack_eval_packer)                                             # unpack eval packer
    session.run(remove_eval_traps)                                              # remove debugger traps in eval statements
    session.run(lift_sequence_expressions)                                      # hoist sequence expressions FIXME exposes transformer bugs
    session.run(unnest_assignment_expressions)                                  # remove nested assignments from returns
    session.run(unflatten_control_flow)                                         # unflatten int machines
    session.run(runtime_global_getter)                                          # find global, inline proxies
    session.run(inline_wrapper_functions)                                       # find runtime function wrappers
    session.run(resolve_wrapper_aliases)                                        # unwrap runtime function call
    session.run(propagate_runtime_globals)                                      # propagate runtime function variables
    session.run(find_runtime_functions)                                         # find and classify runtime functions; wip
    session.run(environmental_scorer, remove_code_locks)                        # remove env scoring calls
    session.run(find_cyclical_array_generators)                                 # generate cyclical array
    session.run(unflatten_control_flow)                                         # unflatten cyclical array machines
    session.run(fold_boolean_comparisons)                                       # resolve cyclical array comparisons
    session.run(restore_local_variables)                                        # extract local variables from array
    session.run(solve_state_machine)                                            # resolve state machine obfuscation
    session.run(fold_constants)                                                 # propagate for inline_aliased_functions
    session.run(find_alias_lookups, inline_alias_lookups)                       # restore obfuscated calls
    session.run(lift_vm)                                                        # lift, restore struture, remove scaffolding
    session.run(solve_string_array)                                             # solve string array, register oracle
    session.run(propagate_runtime_aliases)                                      # propagate runtime aliases within functions FIXME major design flaw
    session.run(fold_constants)                                                 # inline strings
    session.run(resolve_wrapper_aliases)                                        # strings can hide wrapper calls
    session.run(inline_call_wrappers)                                           # inlines more wrapper calls
    session.run_until_stable(fold_constants, simplify_boolean_logic)            # propagate
    session.run(dead_code_elimination)                                          # dce pass
    session.run_until_stable(resolve_hash_guards)                               # resolve nested murmur3 guards
    session.run(remove_dead_code_after_terminator)                              # dce after return/throw/break/continue
    session.run_until_stable(fold_constants, simplify_boolean_logic)            # propagate
    session.run_until_stable(restore_switch_statements, unflatten_control_flow) # resolve nested flattening
    session.run(propagate_aliases)                                              # testing
    session.run(fold_arrays, fold_constants)                                    # propagate string arrays
    session.run(dead_code_elimination)                                          # dce pass
    session.run(restore_for_loops)                                              # restore for loops
    session.run(branches_to_control_flow, lift_sequence_expressions)            # unpack minified control flow
    session.run(inline_trap_functions)                                          # inline runtime trap function TODO remove tamper detection?
    session.run(remove_runtime_submodules)                                      # remove obfuscator scaffolding
    session.run(dead_code_elimination)                                          # dce pass

    # LLM identifier renaming
    #from transformers.rename_vars_llm import LocalLLM, rename_vars_llm
    #llm = LocalLLM(model='qwen2.5-coder', base_url='http://localhost:8080/v1', cache_path='names.cache.jsonl')
    #rename_vars_llm(session, llm, max_prompt_chars=60000)     # for -c 32768

    False and print('\n' + session.report() + '\n')

    print("Saving result: " + str(output_dir / Path(file_path).name))
    with open(output_dir / Path(file_path).name, "w", encoding="utf-8") as file:
        file.write(session.generate())

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.4f} seconds\n")
