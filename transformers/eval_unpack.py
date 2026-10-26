"""
Transformer for the eval/murmur3-keyed XOR packer (the one whose payload
function is named like e_ZdWPD).

    from unpack_eval_packer import unpack_eval_packer
    session.run_until_stable(unpack_eval_packer, ...)

Each run peels ONE layer per eval and splices the plaintext into the tree in
place of the eval. A nested layer comes out as another eval in the next round.

    eval(...);              statement  -> the statements, in place
    return eval(...);       return     -> the statements, last expression returned
    x = eval(...)           expression -> only if the body is all expressions

Anything that cannot be spliced is left as eval("<plaintext>").

The decryption key is derived from the exact text of the packer function inside
the layer string, so this works on the cooked string value (which is what the IR
stores). Anything that changes that string's value breaks the key.

Everything that happens is printed.
"""
import re
import struct
import sys
import urllib.parse
from collections import Counter
from contextlib import contextmanager
from typing import NamedTuple, Optional

import esprima
from jsir import walk
from jsir.bridge import from_esprima
from jsir.tree import node as make
from transformers.helpers.murmur3 import murmur3_x86_32_bytes

# The packer computes this gate as  firefox-UA || !/\n/.test(fn.toString()).
# A shipped payload must decrypt in every browser, so for the original
# (newline-free) function text it is always true. Beautifying a layer adds
# newlines but cannot change what the original gate was, so it stays on.
# If a sample ever decrypts to garbage, this is the first thing to flip.
ANTI_BEAUTIFY_GATE = True

UINT32 = 0xFFFFFFFF
MURMUR_C1, MURMUR_C2 = 0xCC9E2D51, 0x1B873593
KEY_COUNT = 5
PLAUSIBLE_RATIO = 0.9       # fraction of printable ASCII that counts as "decrypted"
PARSE_RECURSION_LIMIT = 20000

# JS \s (ECMAScript WhiteSpace + LineTerminator), not Python's \s
JS_WHITESPACE_CHARS = r'\t\n\v\f\r \u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff'
JS_WHITESPACE = re.compile('[' + JS_WHITESPACE_CHARS + ']+')

# `function name(a, b){ "use strict"; ...` -- the hashed function has this
# directive stripped before hashing, unless the input is a derived key.
STRICT_HEADER = re.compile(
    r'^(function [0-9a-zA-Z_$]+\([0-9a-zA-Z_$]+,[' + JS_WHITESPACE_CHARS + r']*[0-9a-zA-Z_$]+\)'
    r'[' + JS_WHITESPACE_CHARS + r']*\{)'
    r'[' + JS_WHITESPACE_CHARS + r']*("use strict";)(.*)\Z', re.S)


# ------------------------------------------------------- JS emulation
def to_utf16_units(text):
    """str -> list of UTF-16 code units (what charCodeAt sees)."""
    raw = text.encode('utf-16-le', 'surrogatepass')
    return list(struct.unpack('<%dH' % (len(raw) // 2), raw))


def from_utf16_units(units):
    return struct.pack('<%dH' % len(units), *units).decode('utf-16-le', 'surrogatepass')


def murmur3_low_bytes(units, anti_beautify=True):
    """murmur3_x86_32 over the LOW BYTE of each UTF-16 unit, seed == length.

    `anti_beautify` is the packer's gate on the block-mix rotate.
    """
    bytes_data = [unit & 0xFF for unit in units]
    return murmur3_x86_32_bytes(bytes_data, mix_block=anti_beautify)


def hash_key_source(text, anti_beautify=True, is_derived_key=False):
    """Whitespace-insensitive murmur hash of a function's source text."""
    if not is_derived_key:
        text = STRICT_HEADER.sub(r'\1\3', text)
    text = JS_WHITESPACE.sub('', text)
    return murmur3_low_bytes(to_utf16_units(text), anti_beautify)


def decode_uri_component(text):
    if re.search(r'%(?![0-9A-Fa-f]{2})', text):
        raise ValueError('URIError: malformed percent escape')
    return urllib.parse.unquote(text, encoding='utf-8', errors='strict')


def xor_decrypt(cipher_units, seed_key, anti_beautify):
    """Walks the ciphertext back to front. The keystream cycles through
    KEY_COUNT keys, each derived lazily from the previous one."""
    keys = [seed_key]
    key_index = 0
    key_length = len(seed_key)
    position = 0
    plain = []
    for i in range(len(cipher_units) - 1, -1, -1):
        if position == key_length:
            position = 0
            key_index += 1
            if key_index == KEY_COUNT:
                key_index = 0
            if len(keys) < KEY_COUNT:
                previous = keys[key_index - 1]
                keys.append(str(hash_key_source(previous, anti_beautify, is_derived_key=True)))
            key_length = len(keys[key_index])
        plain.append(cipher_units[i] ^ ord(keys[key_index][position]))
        position += 1
    plain.reverse()
    return from_utf16_units(plain)


# ------------------------------------------------------- raw esprima helpers
# These stay on the raw esprima AST on purpose: the key comes from an exact
# source slice of the packer function, which needs ranges into the layer text.
def _raw_walk(node):
    if isinstance(node, list):
        for item in node:
            yield from _raw_walk(item)
    elif hasattr(node, 'type'):
        yield node
        for value in vars(node).values():
            if isinstance(value, list) or hasattr(value, 'type'):
                yield from _raw_walk(value)


def _is_string_literal(node):
    return node is not None and node.type == 'Literal' and isinstance(node.value, str)


@contextmanager
def _deep_recursion():
    """Payloads can nest deeply enough to blow the default recursion limit."""
    previous = sys.getrecursionlimit()
    sys.setrecursionlimit(max(previous, PARSE_RECURSION_LIMIT))
    try:
        yield
    finally:
        sys.setrecursionlimit(previous)


def _complaint_text(error):
    if isinstance(error, dict):
        message = error.get('message', str(error))
    else:
        message = getattr(error, 'description', None) or str(error)
    return re.sub(r'^(Error: )?Line \d+: ', '', message)


def parse_script(text):
    """Tolerant parse with source ranges. -> (ast, tolerated).

    esprima raises some early errors (e.g. "Invalid left-hand side in for-in")
    through a recoverable path; esprima2 reports them on valid code such as
    `for (a[2] in b)`. Tolerant mode returns the AST anyway and lists them in
    `tolerated`, which is what we want: the IR does not care about early errors.
    Real syntax errors still raise."""
    with _deep_recursion():
        ast = esprima.parseScript(text, {'range': True, 'tolerant': True})
    tolerated = [_complaint_text(error) for error in (getattr(ast, 'errors', None) or [])]
    return ast, tolerated


def _try_parse(text):
    """-> (ast, tolerated, failure). `failure` is None when the text parses."""
    try:
        ast, tolerated = parse_script(text)
        return ast, tolerated, None
    except RecursionError:
        return None, [], 'RecursionError while parsing (payload nests too deeply)'
    except Exception as exc:
        message = _complaint_text(exc)
        index = getattr(exc, 'index', None)
        if not isinstance(index, int):
            return None, [], message
        line, column = getattr(exc, 'lineNumber', None), getattr(exc, 'column', None)
        location = f'offset {index}' + (f', line {line} col {column}' if line else '')
        return None, [], f'{message} at {location}, near {text[max(0, index - 40):index + 40]!r}'


def _printable_ratio(text):
    if not text:
        return 0.0
    good = sum(1 for ch in text if ch in '\t\n\r' or 32 <= ord(ch) < 127)
    return good / len(text)


def find_packer_call(layer):
    """-> (packer function source slice, payload string) or None.
    Matches (function name(x){...}('payload')) in the layer text."""
    for node in _raw_walk(parse_script(layer)[0]):
        if (node.type == 'CallExpression' and node.callee.type == 'FunctionExpression'
                and len(node.arguments) == 1 and _is_string_literal(node.arguments[0])
                and len(node.callee.params) == 1):
            start, end = node.callee.range
            return layer[start:end], node.arguments[0].value
    return None


# ------------------------------------------------------- one layer
class LayerResult(NamedTuple):
    text: Optional[str]          # decrypted text, or None if decryption failed
    parses: bool                 # does `text` parse as a script
    reason: Optional[str]        # why it failed, or why it does not parse
    is_packed: bool              # False when the layer is not this packer at all
    ast: object = None           # parsed esprima program for `text`, when it parses
    tolerated: tuple = ()        # parser complaints that were ignored


def unpack_layer(layer):
    try:
        found = find_packer_call(layer)
    except Exception as exc:
        return LayerResult(None, False, f'layer does not parse: {exc}', False)
    if found is None:
        return LayerResult(None, False, 'no packer pattern', False)
    function_source, payload = found

    try:
        cipher = to_utf16_units(decode_uri_component(payload))
    except ValueError as exc:
        return LayerResult(None, False, f'payload is not valid percent-encoding: {exc}', True)

    # the key text has every paren removed, including ones inside strings/regexes
    key_source = function_source.replace('(', '').replace(')', '')
    seed = str(hash_key_source(key_source, ANTI_BEAUTIFY_GATE))
    plain = xor_decrypt(cipher, seed, ANTI_BEAUTIFY_GATE)

    ast, tolerated, failure = _try_parse(plain)
    if failure is None:
        return LayerResult(plain, True, None, True, ast, tuple(tolerated))

    ratio = _printable_ratio(plain)
    if ratio >= PLAUSIBLE_RATIO:
        return LayerResult(plain, False, f'decrypted text does not parse: {failure}', True)
    return LayerResult(None, False,
                       f'decryption produced garbage ({ratio:.0%} printable, '
                       f'starts {plain[:60]!r}); wrong key slice, or the layer text was altered',
                       True)


# ------------------------------------------------------- splicing
def _splice(tree, call, ast):
    """Replace the eval `call` with the program in `ast`. -> None on success,
    otherwise the reason it was not spliced."""
    try:
        with _deep_recursion():
            body = list(from_esprima(ast).root.children_in('body'))
    except RecursionError:
        return 'plaintext is too deeply nested for the IR import (RecursionError)'
    except Exception as exc:
        return f'plaintext does not parse in the IR: {exc}'
    if any(statement.attr('directive') == 'use strict' for statement in body):
        return 'plaintext starts with "use strict"; inlining would change meaning'

    parent = call.parent
    has_lexical = any(
        statement.kind == 'ClassDeclaration' or
        (statement.kind == 'VariableDeclaration' and statement.attr('kind') != 'var')
        for statement in body)

    def put(statements):
        if has_lexical or parent.index_in_parent() is None:
            tree.replace(parent, make('BlockStatement', body=statements))
        else:
            tree.replace_many(parent, statements)

    if parent.kind == 'ExpressionStatement' and parent.get('expression') is call:
        put(body)
        return None

    if parent.kind == 'ReturnStatement' and parent.get('argument') is call:
        if body and body[-1].kind == 'ExpressionStatement':
            returned = make('ReturnStatement', argument=body[-1].get('expression'))
            put(body[:-1] + [returned])
        else:
            # eval's completion value is dropped; only matters if the caller
            # uses the return value of a payload that ends in a non-expression.
            put(body + [make('ReturnStatement', argument=None)])
        return None

    if all(statement.kind == 'ExpressionStatement' for statement in body):
        expressions = [statement.get('expression') for statement in body]
        if not expressions:
            replacement = make('UnaryExpression', operator='void', prefix=True,
                               argument=make('Literal', value=0.0))
        elif len(expressions) == 1:
            replacement = expressions[0]
        else:
            replacement = make('SequenceExpression', expressions=expressions)
        tree.replace(call, replacement)
        return None

    return f'eval is inside a {parent.kind}, which the payload statements cannot replace'


# ------------------------------------------------------- the pass
def unpack_eval_packer(session):
    print("--- Applying Transformer: eval_unpack ---")

    tree = session.tree

    jobs = []
    for call in walk(tree.root, kinds={'CallExpression'}):
        callee = call.get('callee')
        args = call.children_in('arguments')
        if (callee.kind != 'Identifier' or callee.attr('name') != 'eval'
                or session.binding_of(callee) is not None or len(args) != 1):
            continue
        arg = args[0]
        source = arg.attr('value') if arg.kind == 'Literal' else session.constant_of(arg)
        if isinstance(source, str):
            jobs.append((call, arg, source))

    spliced = left_as_string = failed = not_packed = 0
    for call, arg, source in jobs:
        if not call.live:
            continue
        label = f'eval #{call.nid}'
        result = unpack_layer(source)

        if not result.is_packed:
            not_packed += 1
            if result.reason != 'no packer pattern':
                print(f' |-> {label}: not examined, {result.reason}')
            continue

        if result.text is None:
            failed += 1
            print(f'  |-> {label}: FAILED, {result.reason}')
            continue

        if result.tolerated:
            counts = Counter(result.tolerated)
            summary = '; '.join(f'{message} (x{count})' for message, count in counts.items())
            print(f'  |-> {label}: parser tolerated {summary}')

        if not result.parses:
            left_as_string += 1
            print(f'  |-> {label}: decrypted {len(result.text)} chars but {result.reason}')
            print(f'  |-> {label}: left as eval("<plaintext>")')
            tree.replace(arg, make('Literal', value=result.text))
            continue

        where = call.parent.kind
        reason = _splice(tree, call, result.ast)
        if reason is None:
            spliced += 1
            print(f'  |-> {label}: decrypted {len(result.text)} chars, spliced in place of {where}')
        else:
            left_as_string += 1
            print(f'  |-> {label}: decrypted {len(result.text)} chars, '
                  f'left as eval("<plaintext>"): {reason}')
            tree.replace(arg, make('Literal', value=result.text))

    print(f' |-> {len(jobs)} eval(<string>) call(s): {spliced} spliced, '
          f'{left_as_string} left as a string, {failed} failed, '
          f'{not_packed} not this packer')
