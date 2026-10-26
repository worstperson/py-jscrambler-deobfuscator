"""
Node schema.

One table drives traversal, cloning, bridging and printing. The old code had
VISITOR_KEYS plus a reflective fallback over `node.keys()` minus a denylist of
metadata names; when the denylist missed a field, `regex` and `raw` got walked
as if they were children. A node kind that is not in this table is a hard error
rather than a guess.

Each slot is (name, arity, role):
    arity  'one'  a single child or None
           'many' a list of children (entries may be None for array holes)
    role   'value'  evaluated normally
           'target' an assignment/declaration target (a write, not a read)
           'label'  a label name, not a variable reference
           'key'    a property key, only a reference when `computed` is set
"""

ONE = 'one'
MANY = 'many'

VALUE = 'value'
TARGET = 'target'
LABEL = 'label'
KEY = 'key'


def _slots(*spec):
    out = []
    for item in spec:
        if len(item) == 2:
            out.append((item[0], item[1], VALUE))
        else:
            out.append(item)
    return tuple(out)


SCHEMA = {
    'Program': _slots(('body', MANY)),

    # -- statements ---------------------------------------------------------
    'ExpressionStatement': _slots(('expression', ONE)),
    'BlockStatement': _slots(('body', MANY)),
    'EmptyStatement': (),
    'DebuggerStatement': (),
    'ReturnStatement': _slots(('argument', ONE)),
    'IfStatement': _slots(('test', ONE), ('consequent', ONE), ('alternate', ONE)),
    'SwitchStatement': _slots(('discriminant', ONE), ('cases', MANY)),
    'SwitchCase': _slots(('test', ONE), ('consequent', MANY)),
    'ThrowStatement': _slots(('argument', ONE)),
    'TryStatement': _slots(('block', ONE), ('handler', ONE), ('finalizer', ONE)),
    'CatchClause': _slots(('param', ONE, TARGET), ('body', ONE)),
    'WhileStatement': _slots(('test', ONE), ('body', ONE)),
    'DoWhileStatement': _slots(('body', ONE), ('test', ONE)),
    'ForStatement': _slots(('init', ONE), ('test', ONE), ('update', ONE), ('body', ONE)),
    'ForInStatement': _slots(('left', ONE, TARGET), ('right', ONE), ('body', ONE)),
    'ForOfStatement': _slots(('left', ONE, TARGET), ('right', ONE), ('body', ONE)),
    'LabeledStatement': _slots(('label', ONE, LABEL), ('body', ONE)),
    'BreakStatement': _slots(('label', ONE, LABEL)),
    'ContinueStatement': _slots(('label', ONE, LABEL)),
    'WithStatement': _slots(('object', ONE), ('body', ONE)),

    # -- declarations -------------------------------------------------------
    'VariableDeclaration': _slots(('declarations', MANY)),
    'VariableDeclarator': _slots(('id', ONE, TARGET), ('init', ONE)),
    'FunctionDeclaration': _slots(('id', ONE, TARGET), ('params', MANY, TARGET), ('body', ONE)),
    'FunctionExpression': _slots(('id', ONE, TARGET), ('params', MANY, TARGET), ('body', ONE)),
    'ArrowFunctionExpression': _slots(('params', MANY, TARGET), ('body', ONE)),
    'ClassDeclaration': _slots(('id', ONE, TARGET), ('superClass', ONE), ('body', ONE)),
    'ClassExpression': _slots(('id', ONE, TARGET), ('superClass', ONE), ('body', ONE)),
    'ClassBody': _slots(('body', MANY)),
    'MethodDefinition': _slots(('key', ONE, KEY), ('value', ONE)),
    'PropertyDefinition': _slots(('key', ONE, KEY), ('value', ONE)),
    'StaticBlock': _slots(('body', MANY)),

    # -- expressions --------------------------------------------------------
    'Identifier': (),
    'PrivateIdentifier': (),
    'Literal': (),
    'ThisExpression': (),
    'Super': (),
    'ArrayExpression': _slots(('elements', MANY)),
    'ObjectExpression': _slots(('properties', MANY)),
    'Property': _slots(('key', ONE, KEY), ('value', ONE)),
    'ArrayPattern': _slots(('elements', MANY, TARGET)),
    'ObjectPattern': _slots(('properties', MANY, TARGET)),
    'RestElement': _slots(('argument', ONE, TARGET)),
    'SpreadElement': _slots(('argument', ONE)),
    'AssignmentPattern': _slots(('left', ONE, TARGET), ('right', ONE)),
    'SequenceExpression': _slots(('expressions', MANY)),
    'UnaryExpression': _slots(('argument', ONE)),
    'UpdateExpression': _slots(('argument', ONE, TARGET)),
    'BinaryExpression': _slots(('left', ONE), ('right', ONE)),
    'LogicalExpression': _slots(('left', ONE), ('right', ONE)),
    'AssignmentExpression': _slots(('left', ONE, TARGET), ('right', ONE)),
    'ConditionalExpression': _slots(('test', ONE), ('consequent', ONE), ('alternate', ONE)),
    'CallExpression': _slots(('callee', ONE), ('arguments', MANY)),
    'NewExpression': _slots(('callee', ONE), ('arguments', MANY)),
    'MemberExpression': _slots(('object', ONE), ('property', ONE, KEY)),
    'ChainExpression': _slots(('expression', ONE)),
    'TaggedTemplateExpression': _slots(('tag', ONE), ('quasi', ONE)),
    'TemplateLiteral': _slots(('quasis', MANY), ('expressions', MANY)),
    'TemplateElement': (),
    'YieldExpression': _slots(('argument', ONE)),
    'AwaitExpression': _slots(('argument', ONE)),
    'MetaProperty': _slots(('meta', ONE, LABEL), ('property', ONE, LABEL)),

    # -- modules ------------------------------------------------------------
    'ImportDeclaration': _slots(('specifiers', MANY, TARGET), ('source', ONE)),
    'ImportSpecifier': _slots(('imported', ONE, LABEL), ('local', ONE, TARGET)),
    'ImportDefaultSpecifier': _slots(('local', ONE, TARGET)),
    'ImportNamespaceSpecifier': _slots(('local', ONE, TARGET)),
    'ExportNamedDeclaration': _slots(('declaration', ONE), ('specifiers', MANY), ('source', ONE)),
    'ExportSpecifier': _slots(('local', ONE), ('exported', ONE, LABEL)),
    'ExportDefaultDeclaration': _slots(('declaration', ONE)),
    'ExportAllDeclaration': _slots(('source', ONE), ('exported', ONE, LABEL)),
}


# Attributes that are data, not children. Bridged through verbatim.
ATTRIBUTES = {
    'Identifier': ('name',),
    'PrivateIdentifier': ('name',),
    'Literal': ('value',),
    'TemplateElement': ('cooked', 'raw', 'tail'),
    'VariableDeclaration': ('kind',),
    'UnaryExpression': ('operator', 'prefix'),
    'UpdateExpression': ('operator', 'prefix'),
    'BinaryExpression': ('operator',),
    'LogicalExpression': ('operator',),
    'AssignmentExpression': ('operator',),
    'MemberExpression': ('computed', 'optional'),
    'Property': ('kind', 'computed', 'method', 'shorthand'),
    'MethodDefinition': ('kind', 'computed', 'static'),
    'PropertyDefinition': ('computed', 'static'),
    'FunctionDeclaration': ('generator', 'isAsync'),
    'FunctionExpression': ('generator', 'isAsync'),
    'ArrowFunctionExpression': ('generator', 'isAsync', 'expression'),
    'CallExpression': ('optional',),
    'ExpressionStatement': ('directive',),
    'Program': ('sourceType',),
}


FUNCTION_KINDS = frozenset(
    ('FunctionDeclaration', 'FunctionExpression', 'ArrowFunctionExpression'))

LOOP_KINDS = frozenset(
    ('ForStatement', 'ForInStatement', 'ForOfStatement',
     'WhileStatement', 'DoWhileStatement'))

# Kinds that open a `var` scope.
VAR_SCOPE_KINDS = FUNCTION_KINDS | frozenset(('Program', 'StaticBlock'))

# Kinds that open a lexical (let/const/class) scope.
BLOCK_SCOPE_KINDS = VAR_SCOPE_KINDS | frozenset(
    ('BlockStatement', 'ForStatement', 'ForInStatement', 'ForOfStatement',
     'SwitchStatement', 'CatchClause', 'ClassDeclaration', 'ClassExpression'))

STATEMENT_KINDS = frozenset(
    kind for kind in SCHEMA
    if kind.endswith('Statement') or kind in (
        'VariableDeclaration', 'FunctionDeclaration', 'ClassDeclaration',
        'ImportDeclaration', 'ExportNamedDeclaration', 'ExportDefaultDeclaration',
        'ExportAllDeclaration'))


def slots_of(kind):
    try:
        return SCHEMA[kind]
    except KeyError:
        raise KeyError(
            f'unknown node kind {kind!r}: add it to jsir.schema.SCHEMA rather '
            f'than letting traversal guess at its children') from None


def attributes_of(kind):
    return ATTRIBUTES.get(kind, ())
