import math
import struct
from jsir import Session, walk

_RESERVED = frozenset(';/?:@&=+$,#')
_HEX = frozenset('0123456789abcdefABCDEF')


# ---------------------------------------------------------------------------
# JavaScript decoding semantics
# ---------------------------------------------------------------------------

def _hex_byte(text: str, i: int) -> int:
    digits = text[i + 1:i + 3]
    if text[i:i + 1] != '%' or len(digits) != 2 or any(c not in _HEX for c in digits):
        raise ValueError('malformed escape')
    return int(digits, 16)


def _decode_uri(text: str) -> str:
    """ECMAScript decodeURI. Reserved characters (; / ? : @ & = + $ , #) stay
    escaped (unlike Python's unquote); malformed sequences raise ValueError."""
    out = []
    i, size = 0, len(text)
    while i < size:
        if text[i] != '%':
            out.append(text[i])
            i += 1
            continue
        first = _hex_byte(text, i)
        if first < 0x80:
            char = chr(first)
            out.append(text[i:i + 3] if char in _RESERVED else char)
            i += 3
            continue
        if first >> 5 == 0b110:
            extra = 1
        elif first >> 4 == 0b1110:
            extra = 2
        elif first >> 3 == 0b11110:
            extra = 3
        else:
            raise ValueError('bad UTF-8 lead byte')
        chunk = bytearray([first])
        j = i + 3
        for _ in range(extra):
            chunk.append(_hex_byte(text, j))
            j += 3
        try:
            out.append(bytes(chunk).decode('utf-8'))
        except UnicodeDecodeError:
            raise ValueError('bad UTF-8 sequence') from None
        i = j
    return ''.join(out)


def _units(text: str) -> list[int]:
    """Extract 16-bit UTF-16 code units."""
    raw = text.encode('utf-16-le', 'surrogatepass')
    return list(struct.unpack('<%dH' % (len(raw) // 2), raw))


def _text(units: list[int]) -> str:
    """Rebuild string from 16-bit UTF-16 code units."""
    return struct.pack('<%dH' % len(units), *units).decode('utf-16-le', 'surrogatepass')


def js_splice(items: list, start: int, delete_count: int) -> list:
    """Array.prototype.splice: removes in place, returns what was removed."""
    length = len(items)
    begin = max(length + start, 0) if start < 0 else min(start, length)
    count = min(max(delete_count, 0), length - begin)
    removed = items[begin:begin + count]
    del items[begin:begin + count]
    return removed


def apply_shuffles(table: list, ops: list[dict]) -> list:
    """unshift.apply(table, splice(splice(table, o1, l1), o2, l2)), per op in order."""
    table = list(table)
    for op in ops:
        outer = js_splice(table, int(op['s1_offset']), int(op['s1_length']))
        kept = js_splice(outer, int(op['s2_offset']), int(op['s2_length']))
        table = kept + table
    return table


def decode_table(blob: str, xor_key: str, separator: str, ops: list[dict]) -> list[str]:
    key = _units(xor_key)
    if not key:
        raise ValueError('empty XOR key')
    data = _units(_decode_uri(blob))
    mixed = [unit ^ key[index % len(key)] for index, unit in enumerate(data)]
    return apply_shuffles(_text(mixed).split(separator), ops)


def _printable(table: list[str]) -> float:
    chars = ''.join(table)
    return 1.0 if not chars else sum(c.isprintable() for c in chars) / len(chars)


def _lookup(table: list[str]):
    def lookup(index):
        if not isinstance(index, float) or index != int(index) or not 0 <= index < len(table):
            raise IndexError(index)
        return table[int(index)]
    return lookup


class SolvedTable:
    def __init__(self, hit, table, oracle_path):
        self.hit = hit
        self.table = table
        self.oracle_path = oracle_path


class DecoderHit:
    def __init__(self, node, call_node, captures):
        self.node = node
        self.region = node
        self.call_node = call_node
        self.captures = captures

    def __getitem__(self, key):
        return self.captures[key]

    def unclaimed_effects(self):
        return []


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def get_args(call_node) -> list:
    if not call_node:
        return []
    if hasattr(call_node, 'children_in'):
        try:
            res = list(call_node.children_in('arguments'))
            if res:
                return res
        except Exception:
            pass
    args = call_node.get('arguments')
    return list(args) if args else []


def get_constant(session, node):
    if node is None:
        return None
    if node.kind == 'Literal':
        return node.attr('value')
    if node.kind == 'UnaryExpression' and node.attr('operator') in ('-', '+'):
        inner = node.get('argument')
        if inner and inner.kind == 'Literal' and isinstance(inner.attr('value'), (int, float)) and not isinstance(inner.attr('value'), bool):
            val = float(inner.attr('value'))
            return -val if node.attr('operator') == '-' else val
    if hasattr(session, 'constant_of'):
        return session.constant_of(node)
    return None


def get_numeric_value(session, node) -> float:
    val = get_constant(session, node)
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    return float('nan')


def get_member_name(node) -> str | None:
    if not node or node.kind != 'MemberExpression':
        return None
    prop = node.get('property')
    if not prop:
        return None
    if node.attr('computed', False):
        val = prop.attr('value')
        return str(val) if prop.kind == 'Literal' else None
    return prop.attr('name') if prop.kind == 'Identifier' else None


def find_ancestor(node, predicate):
    curr = getattr(node, 'parent', None)
    while curr is not None:
        if predicate(curr):
            return curr
        curr = getattr(curr, 'parent', None)
    return None


def _find_runtime_slot(call, base_name):
    curr = call.parent
    while curr is not None:
        if curr.kind == 'AssignmentExpression' and curr.attr('operator', '=') == '=':
            left = curr.get('left')
            if left and left.kind == 'MemberExpression':
                obj, prop = left.get('object'), left.get('property')
                if obj and obj.kind == 'Identifier' and prop is not None:
                    if obj.attr('name') == base_name:
                        if prop.kind == 'Literal' and isinstance(prop.attr('value'), (int, float)):
                            return curr, str(int(prop.attr('value')))
                        if prop.kind == 'Literal' and isinstance(prop.attr('value'), str):
                            return curr, prop.attr('value')
                        if prop.kind == 'Identifier' and not left.attr('computed', False):
                            return curr, prop.attr('name')
        curr = curr.parent
    return None, None


def find_registration(call, base_name):
    # 1. Nested in ObjectProperty: _global[slot] = { prop: (function(...) { ... })() }
    prop_node = find_ancestor(call, lambda n: n.kind in ('Property', 'ObjectProperty'))
    if prop_node:
        key_node = prop_node.get('key')
        prop_key = (key_node.attr('name') if key_node and key_node.kind == 'Identifier'
                    else str(key_node.attr('value')) if key_node else None)
        slot_node, slot_key = _find_runtime_slot(call, base_name)
        if slot_key and prop_key:
            return {'assignment_node': slot_node, 'slot_key': slot_key, 'prop_key': prop_key, 'is_nested': True}

    # 2. Direct property assignment: _global[slot].prop = (function(...) { ... })()
    curr = call.parent
    while curr is not None:
        if curr.kind == 'AssignmentExpression' and curr.attr('operator', '=') == '=':
            left = curr.get('left')
            if left and left.kind == 'MemberExpression':
                inner = left.get('object')
                if inner and inner.kind == 'MemberExpression':
                    g_obj, g_prop = inner.get('object'), inner.get('property')
                    if g_obj and g_obj.kind == 'Identifier' and g_obj.attr('name') == base_name:
                        if g_prop:
                            slot_key = (str(int(g_prop.attr('value'))) if g_prop.kind == 'Literal' and isinstance(g_prop.attr('value'), (int, float))
                                        else g_prop.attr('name') if g_prop.kind == 'Identifier' and not inner.attr('computed', False) else None)
                            outer_prop = left.get('property')
                            prop_key = (outer_prop.attr('name') if outer_prop and outer_prop.kind == 'Identifier' and not left.attr('computed', False)
                                        else str(outer_prop.attr('value')) if outer_prop and outer_prop.kind == 'Literal' else None)
                            if slot_key and prop_key:
                                return {'assignment_node': curr, 'slot_key': slot_key, 'prop_key': prop_key, 'is_nested': False}
        curr = curr.parent

    slot_node, slot_key = _find_runtime_slot(call, base_name)
    return {'assignment_node': slot_node, 'slot_key': slot_key, 'prop_key': None, 'is_nested': False}


# ---------------------------------------------------------------------------
# Decoder feature extraction
# ---------------------------------------------------------------------------

def has_xor_loop(func_node) -> bool:
    for loop in walk(func_node, kinds={'WhileStatement', 'DoWhileStatement', 'ForStatement'}, enter_functions=False):
        for be in walk(loop, kinds={'BinaryExpression'}, enter_functions=False):
            if be.attr('operator') == '^':
                return True
        for ae in walk(loop, kinds={'AssignmentExpression'}, enter_functions=False):
            if ae.attr('operator') == '^=':
                return True
    return False


def find_separator(session, func_node) -> str | None:
    for call in walk(func_node, kinds={'CallExpression'}, enter_functions=False):
        callee = call.get('callee')
        args = get_args(call)
        if not callee:
            continue
        if callee.kind == 'MemberExpression' and get_member_name(callee) == 'split':
            if len(args) >= 1:
                c = get_constant(session, args[0])
                if isinstance(c, str):
                    return c
        if callee.kind == 'MemberExpression' and get_member_name(callee) == 'call':
            obj = callee.get('object')
            if obj and get_member_name(obj) == 'split' and len(args) >= 2:
                c = get_constant(session, args[1])
                if isinstance(c, str):
                    return c
        if (callee.kind == 'Identifier' and callee.attr('name') == 'split') or \
           (callee.kind == 'MemberExpression' and get_member_name(callee) == 'split'):
            if len(args) >= 2:
                c = get_constant(session, args[1])
                if isinstance(c, str):
                    return c
    return None


def get_incremented_identifier(session, node) -> str | None:
    if node.kind == 'UpdateExpression' and node.attr('operator') == '++':
        arg = node.get('argument')
        if arg and arg.kind == 'Identifier':
            return arg.attr('name')
    if node.kind == 'AssignmentExpression':
        left = node.get('left')
        if not left or left.kind != 'Identifier':
            return None
        var_name = left.attr('name')
        op = node.attr('operator')
        right = node.get('right')
        if op == '+=' and get_numeric_value(session, right) == 1.0:
            return var_name
        if op == '=' and right and right.kind == 'BinaryExpression' and right.attr('operator') == '+':
            rl, rr = right.get('left'), right.get('right')
            if rl and rl.kind == 'Identifier' and rl.attr('name') == var_name and get_numeric_value(session, rr) == 1.0:
                return var_name
            if rr and rr.kind == 'Identifier' and rr.attr('name') == var_name and get_numeric_value(session, rl) == 1.0:
                return var_name
    return None


def is_splice_callee(node) -> bool:
    if not node:
        return False
    if node.kind == 'Identifier' and node.attr('name') == 'splice':
        return True
    if node.kind == 'MemberExpression':
        prop = node.get('property')
        if prop and prop.kind == 'Identifier' and prop.attr('name') == 'splice':
            return True
        if prop and prop.kind == 'Literal' and prop.attr('value') == 'splice':
            return True
    return False


def extract_splice_op_from_call(session, call_node):
    if call_node.kind != 'CallExpression':
        return None
    callee = call_node.get('callee')
    args = get_args(call_node)

    # Case A: table.splice(s1_o, s1_l).splice(s2_o, s2_l)
    if callee and callee.kind == 'MemberExpression' and is_splice_callee(callee):
        inner = callee.get('object')
        if inner and inner.kind == 'CallExpression' and is_splice_callee(inner.get('callee')):
            inner_args = get_args(inner)
            if len(inner_args) >= 2 and len(args) >= 2:
                s1_off = get_numeric_value(session, inner_args[0])
                s1_len = get_numeric_value(session, inner_args[1])
                s2_off = get_numeric_value(session, args[0])
                s2_len = get_numeric_value(session, args[1])
                if not any(math.isnan(v) for v in (s1_off, s1_len, s2_off, s2_len)):
                    return s1_off, s1_len, s2_off, s2_len

    # Case B: splice(splice(table, s1_o, s1_l), s2_o, s2_l) or _global.splice(...)
    if is_splice_callee(callee) and len(args) >= 3:
        inner = args[0]
        if inner and inner.kind == 'CallExpression' and is_splice_callee(inner.get('callee')):
            inner_args = get_args(inner)
            if len(inner_args) >= 3:
                s1_off = get_numeric_value(session, inner_args[1])
                s1_len = get_numeric_value(session, inner_args[2])
                s2_off = get_numeric_value(session, args[1])
                s2_len = get_numeric_value(session, args[2])
                if not any(math.isnan(v) for v in (s1_off, s1_len, s2_off, s2_len)):
                    return s1_off, s1_len, s2_off, s2_len

    return None


def extract_equality_pairs(session, node):
    if node is None:
        return []
    if node.kind == 'LogicalExpression' and node.attr('operator') == '&&':
        return extract_equality_pairs(session, node.get('left')) + extract_equality_pairs(session, node.get('right'))
    if node.kind == 'BinaryExpression' and node.attr('operator') in ('===', '=='):
        l, r = node.get('left'), node.get('right')
        l_num = get_numeric_value(session, l)
        r_num = get_numeric_value(session, r)
        if not math.isnan(l_num) and r and r.kind == 'Identifier':
            return [(r.attr('name'), l_num)]
        if not math.isnan(r_num) and l and l.kind == 'Identifier':
            return [(l.attr('name'), r_num)]
    return []


def extract_rotate_ops(session, func_node) -> list[dict]:
    raw_ops = []
    # Enter nested functions because state machine branches are often inside a helper function (e.g., N0)
    for if_node in walk(func_node, kinds={'IfStatement'}):
        test = if_node.get('test')
        pairs = extract_equality_pairs(session, test)
        if len(pairs) < 2:
            continue

        consequent = if_node.get('consequent')
        if not consequent:
            continue

        incr_vars = set()
        for n in walk(consequent, enter_functions=False):
            iv = get_incremented_identifier(session, n)
            if iv:
                incr_vars.add(iv)

        splice_op = None
        for n in walk(consequent, kinds={'CallExpression'}, enter_functions=False):
            sp = extract_splice_op_from_call(session, n)
            if sp:
                splice_op = sp
                break
        if not splice_op:
            continue

        for var_name, num_val in pairs:
            if var_name in incr_vars:
                state_var = var_name
                step = num_val
                for other_var, other_num in pairs:
                    if other_var != state_var:
                        raw_ops.append({
                            'state_var': state_var,
                            'arg_var': other_var,
                            'step': step,
                            'magic': other_num,
                            's1_offset': splice_op[0],
                            's1_length': splice_op[1],
                            's2_offset': splice_op[2],
                            's2_length': splice_op[3],
                        })
                        break
                break

    from collections import defaultdict
    groups = defaultdict(list)
    for op in raw_ops:
        groups[(op['state_var'], op['arg_var'])].append(op)

    best_ops = []
    for _key, ops in groups.items():
        ops_sorted = sorted(ops, key=lambda x: x['step'])
        steps = [o['step'] for o in ops_sorted]
        if steps == [float(i) for i in range(len(steps))] and len(steps) > len(best_ops):
            best_ops = ops_sorted

    return best_ops


def find_xor_key(session, func_node, call_node) -> str | None:
    params = list(func_node.children_in('params')) if hasattr(func_node, 'children_in') else (func_node.get('params') or [])
    param_names = [p.attr('name') for p in params if p and p.kind == 'Identifier']
    args = get_args(call_node)

    # 1. Match direct charCodeAt or charCodeAt.bind on a parameter
    for call in walk(func_node, kinds={'CallExpression'}, enter_functions=False):
        callee = call.get('callee')
        call_args = get_args(call)
        target_name = None
        if callee and callee.kind == 'MemberExpression':
            if get_member_name(callee) == 'charCodeAt':
                obj = callee.get('object')
                if obj and obj.kind == 'Identifier':
                    target_name = obj.attr('name')
            elif get_member_name(callee) == 'bind':
                bound_obj = callee.get('object')
                if bound_obj and bound_obj.kind == 'MemberExpression' and get_member_name(bound_obj) == 'charCodeAt':
                    obj = bound_obj.get('object')
                    if obj and obj.kind == 'Identifier':
                        target_name = obj.attr('name')
        elif len(call_args) >= 1:
            first = call_args[0]
            if first and first.kind == 'Identifier':
                if (callee and callee.kind == 'Identifier' and callee.attr('name') == 'charCodeAt') or \
                   (callee and callee.kind == 'MemberExpression' and get_member_name(callee) == 'charCodeAt'):
                    target_name = first.attr('name')

        if target_name and target_name in param_names:
            idx = param_names.index(target_name)
            if idx < len(args):
                c = get_constant(session, args[idx])
                if isinstance(c, str):
                    return c

    # 2. Fallback to the first string constant argument passed to the IIFE
    for arg in args:
        c = get_constant(session, arg)
        if isinstance(c, str):
            return c
    return None


def find_decoders(session) -> list[DecoderHit]:
    tree = session.tree
    base_name = getattr(session, 'runtime_object', None)
    decoders = []

    for call in walk(tree.root, kinds={'CallExpression'}):
        callee = call.get('callee')
        if callee and callee.kind == 'SequenceExpression':
            exprs = list(callee.children_in('expressions')) if hasattr(callee, 'children_in') else (callee.get('expressions') or [])
            if exprs:
                callee = exprs[-1]

        if not callee or callee.kind not in ('FunctionExpression', 'ArrowFunctionExpression'):
            continue

        if not has_xor_loop(callee):
            continue

        separator = find_separator(session, callee)
        if separator is None:
            continue

        ops = extract_rotate_ops(session, callee)
        if not ops:
            continue

        xor_key = find_xor_key(session, callee, call)
        if not xor_key or not isinstance(xor_key, str):
            continue

        reg = find_registration(call, base_name)
        if not reg or not reg.get('slot_key') or not reg.get('prop_key'):
            continue

        captures = {
            'xor_key': xor_key,
            'separator': separator,
            'ops': ops,
            'registration': reg,
        }
        decoders.append(DecoderHit(callee, call, captures))

    return decoders


# ---------------------------------------------------------------------------
# String functions and transformer entrypoint
# ---------------------------------------------------------------------------

def find_large_string_functions(tree, minimum: int = 50) -> list[dict]:
    found = []

    def check_fn(func, name):
        body = func.get('body')
        if not body:
            return
        if body.kind == 'Literal':
            val = body.attr('value')
            if isinstance(val, str) and len(val) >= minimum:
                found.append({'value': val, 'node': func, 'name': name})
            return
        if body.kind != 'BlockStatement':
            return
        statements = list(body.children_in('body')) if hasattr(body, 'children_in') else (body.get('body') or [])
        if len(statements) != 1 or statements[0].kind != 'ReturnStatement':
            return
        arg = statements[0].get('argument')
        if not arg or arg.kind != 'Literal':
            return
        val = arg.attr('value')
        if isinstance(val, str) and len(val) >= minimum:
            found.append({'value': val, 'node': func, 'name': name})

    for func in walk(tree.root, kinds={'FunctionDeclaration'}):
        ident = func.get('id')
        name = ident.attr('name') if ident is not None else '(anonymous)'
        check_fn(func, name)

    for assign in walk(tree.root, kinds={'AssignmentExpression'}):
        right = assign.get('right')
        if right and right.kind in ('FunctionExpression', 'ArrowFunctionExpression'):
            left = assign.get('left')
            name = '(anonymous)'
            if left:
                if left.kind == 'Identifier':
                    name = left.attr('name')
                elif left.kind == 'MemberExpression':
                    prop = left.get('property')
                    name = prop.attr('name') if prop and prop.kind == 'Identifier' else str(prop.attr('value')) if prop else '(anonymous)'
            check_fn(right, name)

    return found


def solve_string_tables(session: Session, minimum_printable: float = 0.9) -> list[SolvedTable]:
    decoders = find_decoders(session)
    if not decoders:
        print(' |-> Failed to find any string decoder functions.')
        return []

    blobs = find_large_string_functions(session.tree)
    if not blobs:
        print(' |-> Found string decoder(s) but no encoded string function.')
        return []

    solved = []
    for hit in decoders:
        captures = hit.captures
        registration = captures['registration'] or {}
        if captures['xor_key'] is None or not isinstance(captures['xor_key'], str):
            print(' |-> Decoder has no constant XOR key at its call site; skipping.')
            continue
        if registration.get('slot_key') is None or registration.get('prop_key') is None:
            print(' |-> Decoder is not published as `_global[<slot>].<name>`; skipping.')
            continue

        scored = []
        for blob in blobs:
            try:
                table = decode_table(blob['value'], captures['xor_key'],
                                     captures['separator'], captures['ops'])
            except (ValueError, IndexError):
                continue
            scored.append((_printable(table), blob, table))

        scored.sort(key=lambda item: -item[0])
        if not scored or scored[0][0] < minimum_printable or \
                (len(scored) > 1 and scored[1][0] >= minimum_printable):
            print(f' |-> Could not pair decoder {registration["slot_key"]}.{registration["prop_key"]} '
                  f'with exactly one encoded string function.')
            continue

        _ratio, blob, table = scored[0]
        path = f'{session.runtime_object}.{registration["slot_key"]}.{registration["prop_key"]}'
        print(f' |-> Decoder {path}: key {captures["xor_key"]!r}, separator {captures["separator"]!r}, '
              f'{len(captures["ops"])} shuffles, blob "{blob["name"]}", {len(table)} strings')

        session.oracle.calls[path] = _lookup(table)
        if hasattr(session, 'string_functions'):
            session.string_functions.append(blob['node'])
        solved.append(SolvedTable(hit, table, path))

    return solved


def solve_string_array(session: Session) -> None:
    print("--- Applying Transformer: solve_string_array ---")
    solve_string_tables(session)
