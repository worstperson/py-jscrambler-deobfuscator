"""
Monkey patch for escodegen to add AST generation support for ES2022 Class
Static Blocks (StaticBlock), Private Identifiers (PrivateIdentifier), and
Class Fields (PropertyDefinition).
"""

from escodegen import escodegen


escodegen.Syntax.StaticBlock = "StaticBlock"


def PropertyDefinition(self, expr, precedence, flags):
    result = ['static' + escodegen.noEmptySpace()] if expr.static else []
    result.append(self.generatePropertyKey(expr.key, expr.computed))
    if expr.value:
        result.append(escodegen.space + '=' + escodegen.space)
        result.append(self.generateExpression(expr.value, escodegen.Precedence.Assignment, escodegen.E_TTT))
    # Always terminate: fields have no ASI-safe form once methods or
    # computed keys follow them.
    result.append(';')
    return result

def StaticBlock(self, expr, precedence, flags):
    return ['static' + escodegen.space, self.BlockStatement(expr, escodegen.S_TFFF)]


def PrivateIdentifier(self, expr, precedence, flags):
    return expr.name


escodegen.CodeGeneratorExpression.PropertyDefinition = PropertyDefinition
escodegen.CodeGeneratorExpression.StaticBlock = StaticBlock
escodegen.CodeGeneratorExpression.PrivateIdentifier = PrivateIdentifier
escodegen.CodeGeneratorStatement.StaticBlock = StaticBlock


"""
Monkey patch for escodegen to prevent bare literals generating in MemberExpressions
"""


def _MemberExpression(self, expr, precedence, flags):
    result = [self.generateExpression(expr.object, escodegen.Precedence.Call,
            escodegen.E_TTF if (flags & escodegen.F_ALLOW_CALL) else escodegen.E_TFF)]

    if expr.computed:
        if expr.optional:
            result.append('?.')
        result.append('[')
        result.append(self.generateExpression(expr.property, escodegen.Precedence.Sequence,
                escodegen.E_TTT if (flags & escodegen.F_ALLOW_CALL) else escodegen.E_TFT))
        result.append(']')
    else:
        if not expr.optional and expr.object.type == escodegen.Syntax.Literal and \
                isinstance(expr.object.value, (int, float)):
            # Uniform, regardless of whether the tokenizer would have been
            # fine with a bare literal here.
            result = ['('] + result + [')']

        result.append('?.' if expr.optional else '.')
        result.append(escodegen.generateIdentifier(expr.property))

    return escodegen.parenthesize(result, escodegen.Precedence.Member, precedence)


escodegen.CodeGeneratorExpression.MemberExpression = _MemberExpression
