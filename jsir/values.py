"""
The value domain.

Two decisions here that the old code got wrong, both of which caused real bugs:

1. `null` is not Python `None`. In the old passes `get(node, 'value')` returned
   `None` both for a `null` literal and for a field that was absent, and
   `resolve_identifier` returned `None` for "no value". Three meanings on one
   object. Here `null` is NULL, `undefined` is UNDEF, and absence is Python
   `None` and never reaches the value domain.

2. Numbers are doubles. Every JS number is a float; there is no integer type.
   The old code carried Python ints through and then had to write
   `isinstance(v, (int, float)) and not isinstance(v, bool)` at a dozen call
   sites, which is both noise and a bool/int trap waiting to fire. Import
   normalises every numeric literal to float, once.

Lattice
-------
The names `top` and `bottom` flip meaning between textbooks and are a reliable
source of inverted-join bugs, so they are not used:

    UNSET     no value has reached this point yet (optimistic start, and the
              value of an unreachable edge)
    UNKNOWN   more than one value reaches here, or we cannot compute it

    join(UNSET, x)   == x
    join(x, x)       == x
    join(x, y)       == UNKNOWN     for x != y
    join(UNKNOWN, x) == UNKNOWN
"""

import math
import re


class _Singleton:
    __slots__ = ('_name',)
    _registry = {}

    def __new__(cls, name):
        if name not in cls._registry:
            instance = super().__new__(cls)
            instance._name = name
            cls._registry[name] = instance
        return cls._registry[name]

    def __repr__(self):
        return self._name

    def __bool__(self):
        raise TypeError(f'{self._name} has no Python truth value; use truthiness()')


NULL = _Singleton('null')
UNDEF = _Singleton('undefined')
UNSET = _Singleton('UNSET')
UNKNOWN = _Singleton('UNKNOWN')

LATTICE_MARKERS = (UNSET, UNKNOWN)


class JSRegExp:
    __slots__ = ('pattern', 'flags')

    def __init__(self, pattern, flags):
        self.pattern = pattern
        self.flags = flags

    def __eq__(self, other):
        return (isinstance(other, JSRegExp) and other.pattern == self.pattern
                and other.flags == self.flags)

    def __hash__(self):
        return hash((self.pattern, self.flags))

    def __repr__(self):
        return f'/{self.pattern}/{self.flags}'


class Opaque:
    """A value we can identify but not read: a host object, a function, an
    object whose contents escaped. Distinct from UNKNOWN because identity and
    truthiness are still known, which is enough to decide `x && ...`."""

    __slots__ = ('label', 'truthy')

    def __init__(self, label, truthy=True):
        self.label = label
        self.truthy = truthy

    def __repr__(self):
        return f'<{self.label}>'


def is_concrete(value):
    """True for values that can be written back into the tree as a literal."""
    return (value is NULL or value is UNDEF or
            isinstance(value, (bool, float, str, JSRegExp)))


def is_resolved(value):
    """True for anything that is a value at all, opaque ones included."""
    return value not in LATTICE_MARKERS


def join(left, right):
    if left is UNSET:
        return right
    if right is UNSET:
        return left
    if left is UNKNOWN or right is UNKNOWN:
        return UNKNOWN
    if left is right:
        return left
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) and math.isnan(right):
            return left
        return left if left == right else UNKNOWN
    if type(left) is not type(right):
        return UNKNOWN
    return left if left == right else UNKNOWN


# ---------------------------------------------------------------------------
# Coercions
# ---------------------------------------------------------------------------

_HEX = re.compile(r'^[+-]?0[xX][0-9a-fA-F]+$')
_OCT = re.compile(r'^[+-]?0[oO][0-7]+$')
_BIN = re.compile(r'^[+-]?0[bB][01]+$')


def truthiness(value):
    """True, False, or UNKNOWN. Never raises."""
    if value in LATTICE_MARKERS:
        return UNKNOWN
    if value is NULL or value is UNDEF:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return not (value == 0.0 or math.isnan(value))
    if isinstance(value, str):
        return len(value) > 0
    if isinstance(value, Opaque):
        return value.truthy
    return True


def to_number(value):
    if value in LATTICE_MARKERS or isinstance(value, Opaque):
        return UNKNOWN
    if value is UNDEF:
        return math.nan
    if value is NULL:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, float):
        return value
    if isinstance(value, JSRegExp):
        return math.nan
    text = value.strip()
    if text == '':
        return 0.0
    try:
        if _HEX.match(text):
            return float(int(text, 16))
        if _OCT.match(text):
            return float(int(text, 8))
        if _BIN.match(text):
            return float(int(text, 2))
        low = text.lower()
        if low in ('infinity', '+infinity'):
            return math.inf
        if low == '-infinity':
            return -math.inf
        if low == 'nan':
            return math.nan
        return float(text)
    except ValueError:
        return math.nan


def number_to_string(number):
    """JS Number::toString. Not repr(); `1.0` prints as "1"."""
    if math.isnan(number):
        return 'NaN'
    if math.isinf(number):
        return 'Infinity' if number > 0 else '-Infinity'
    if number == 0.0:
        return '0'
    if number == int(number) and abs(number) < 1e21:
        return str(int(number))
    text = repr(number)
    if 'e' in text:
        mantissa, exponent = text.split('e')
        sign = '+' if not exponent.startswith('-') else '-'
        return f"{mantissa}e{sign}{exponent.lstrip('+-').lstrip('0') or '0'}"
    return text


def to_string(value):
    if value in LATTICE_MARKERS or isinstance(value, Opaque):
        return UNKNOWN
    if value is UNDEF:
        return 'undefined'
    if value is NULL:
        return 'null'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, float):
        return number_to_string(value)
    if isinstance(value, JSRegExp):
        return f'/{value.pattern}/{value.flags}'
    return value


def to_int32(value):
    number = to_number(value)
    if number is UNKNOWN or math.isnan(number) or math.isinf(number):
        return 0 if number is not UNKNOWN else UNKNOWN
    bits = int(math.trunc(number)) & 0xFFFFFFFF
    return bits - 0x100000000 if bits >= 0x80000000 else bits


def to_uint32(value):
    number = to_number(value)
    if number is UNKNOWN:
        return UNKNOWN
    if math.isnan(number) or math.isinf(number):
        return 0
    return int(math.trunc(number)) & 0xFFFFFFFF


def type_of(value):
    if value in LATTICE_MARKERS:
        return UNKNOWN
    if value is UNDEF:
        return 'undefined'
    if value is NULL:
        return 'object'
    if isinstance(value, bool):
        return 'boolean'
    if isinstance(value, float):
        return 'number'
    if isinstance(value, str):
        return 'string'
    if isinstance(value, Opaque):
        return 'function' if value.label.startswith('fn:') else 'object'
    return 'object'


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

def strict_equals(left, right):
    if left in LATTICE_MARKERS or right in LATTICE_MARKERS:
        return UNKNOWN
    if isinstance(left, Opaque) or isinstance(right, Opaque):
        return left is right
    if (left is NULL) != (right is NULL) or (left is UNDEF) != (right is UNDEF):
        return False
    if left is NULL or left is UNDEF:
        return True
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    if isinstance(left, bool):
        return left == right
    if isinstance(left, float) and isinstance(right, float):
        return not (math.isnan(left) or math.isnan(right)) and left == right
    if isinstance(left, str) and isinstance(right, str):
        return left == right
    if isinstance(left, JSRegExp) or isinstance(right, JSRegExp):
        return left is right      # object identity
    return False


def loose_equals(left, right):
    if left in LATTICE_MARKERS or right in LATTICE_MARKERS:
        return UNKNOWN
    nullish_left = left is NULL or left is UNDEF
    nullish_right = right is NULL or right is UNDEF
    if nullish_left or nullish_right:
        return nullish_left and nullish_right
    if isinstance(left, Opaque) or isinstance(right, Opaque):
        return left is right if isinstance(left, Opaque) and isinstance(right, Opaque) else UNKNOWN
    if isinstance(left, bool):
        return loose_equals(1.0 if left else 0.0, right)
    if isinstance(right, bool):
        return loose_equals(left, 1.0 if right else 0.0)
    if isinstance(left, JSRegExp) or isinstance(right, JSRegExp):
        return UNKNOWN
    if isinstance(left, float) and isinstance(right, float):
        return strict_equals(left, right)
    if isinstance(left, str) and isinstance(right, str):
        return left == right
    if isinstance(left, float) and isinstance(right, str):
        return strict_equals(left, to_number(right))
    if isinstance(left, str) and isinstance(right, float):
        return strict_equals(to_number(left), right)
    return UNKNOWN


def _shift_guard(*parts):
    return any(part is UNKNOWN for part in parts)


def binary(operator, left, right):
    """Apply a binary operator. Returns a value or UNKNOWN."""
    if left in LATTICE_MARKERS or right in LATTICE_MARKERS:
        return UNKNOWN

    if operator == '===':
        return _negate(strict_equals(left, right), False)
    if operator == '!==':
        return _negate(strict_equals(left, right), True)
    if operator == '==':
        return _negate(loose_equals(left, right), False)
    if operator == '!=':
        return _negate(loose_equals(left, right), True)
    if operator in ('in', 'instanceof'):
        return UNKNOWN

    if isinstance(left, Opaque) or isinstance(right, Opaque):
        return UNKNOWN

    if operator == '+':
        if isinstance(left, str) or isinstance(right, str) or \
                isinstance(left, JSRegExp) or isinstance(right, JSRegExp):
            left_text, right_text = to_string(left), to_string(right)
            if left_text is UNKNOWN or right_text is UNKNOWN:
                return UNKNOWN
            return left_text + right_text
        return _arith('+', left, right)

    if operator in ('-', '*', '/', '%', '**'):
        return _arith(operator, left, right)

    if operator in ('&', '|', '^'):
        a, b = to_int32(left), to_int32(right)
        if _shift_guard(a, b):
            return UNKNOWN
        result = {'&': a & b, '|': a | b, '^': a ^ b}[operator]
        return float(_wrap32(result))
    if operator == '<<':
        a, b = to_int32(left), to_uint32(right)
        if _shift_guard(a, b):
            return UNKNOWN
        return float(_wrap32(a << (b & 31)))
    if operator == '>>':
        a, b = to_int32(left), to_uint32(right)
        if _shift_guard(a, b):
            return UNKNOWN
        return float(a >> (b & 31))
    if operator == '>>>':
        a, b = to_uint32(left), to_uint32(right)
        if _shift_guard(a, b):
            return UNKNOWN
        return float(a >> (b & 31))

    if operator in ('<', '<=', '>', '>='):
        if isinstance(left, str) and isinstance(right, str):
            a, b = left, right
        else:
            a, b = to_number(left), to_number(right)
            if a is UNKNOWN or b is UNKNOWN:
                return UNKNOWN
            if math.isnan(a) or math.isnan(b):
                return False
        return {'<': a < b, '<=': a <= b, '>': a > b, '>=': a >= b}[operator]

    return UNKNOWN


def _negate(result, flip):
    if result is UNKNOWN:
        return UNKNOWN
    return (not result) if flip else result


def _wrap32(value):
    bits = value & 0xFFFFFFFF
    return bits - 0x100000000 if bits >= 0x80000000 else bits


def _arith(operator, left, right):
    a, b = to_number(left), to_number(right)
    if a is UNKNOWN or b is UNKNOWN:
        return UNKNOWN
    try:
        if operator == '+':
            return a + b
        if operator == '-':
            return a - b
        if operator == '*':
            return a * b
        if operator == '/':
            if b == 0.0:
                if math.isnan(a) or a == 0.0:
                    return math.nan
                negative = (a < 0) != (math.copysign(1.0, b) < 0)
                return -math.inf if negative else math.inf
            return a / b
        if operator == '%':
            if b == 0.0 or math.isnan(a) or math.isnan(b) or math.isinf(a):
                return math.nan
            if math.isinf(b):
                return a
            return math.fmod(a, b)
        if operator == '**':
            return float(a ** b)
    except (OverflowError, ValueError, ZeroDivisionError):
        return math.nan
    return UNKNOWN


def unary(operator, value):
    if operator == 'void':
        return UNDEF if value is not UNSET else UNSET
    if value in LATTICE_MARKERS:
        return UNKNOWN
    if operator == 'typeof':
        return type_of(value)
    if operator == '!':
        truth = truthiness(value)
        return UNKNOWN if truth is UNKNOWN else (not truth)
    if isinstance(value, Opaque):
        return UNKNOWN
    if operator == '+':
        return to_number(value)
    if operator == '-':
        number = to_number(value)
        return UNKNOWN if number is UNKNOWN else -number
    if operator == '~':
        bits = to_int32(value)
        return UNKNOWN if bits is UNKNOWN else float(_wrap32(~bits))
    return UNKNOWN
