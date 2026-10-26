from types import FunctionType, BuiltinFunctionType, MethodType, BuiltinMethodType
import six
if six.PY3:
    basestring = str
    long = int

# https://github.com/Marven11/CVE-2024-28397-js2py-Sandbox-Escape/blob/main/fix.py
from js2py.constructors.jsobject import Object
fn = Object.own["getOwnPropertyNames"]["value"].code
def wraps(*args, **kwargs):
    result = fn(*args, **kwargs)
    return list(result)
Object.own["getOwnPropertyNames"]["value"].code = wraps

# https://github.com/PiotrDabkowski/Js2Py/pull/323#issuecomment-3837280531
def _py_wrap(py):
    if isinstance(py, (FunctionType, BuiltinFunctionType, MethodType,
                       BuiltinMethodType, dict, int, str, bool, float, list,
                       tuple, long, basestring, bytearray, bytes)) or py is None:
        return js2py.base.HJs(py)
    return js2py.base.PyObjectWrapper(py)
js2py.base.py_wrap = _py_wrap
