"""
Monkey patch for esprima.scanner.Scanner to support legacy (Annex B) octal
escapes in non-unicode regular expression literals (e.g., /\\105\\162/).
"""

import sys
import re
from esprima.scanner import Scanner, Character


def readLegacyRegExpOctalEscape(self, pattern, index):
    length = len(pattern)
    end = index + 2
    maxEnd = index + (4 if pattern[index + 1] in '0123' else 3)
    while end < length and end < maxEnd and pattern[end] in '01234567':
        end += 1
    return int(pattern[index + 1:end], 8), end


def sanitizeLegacyRegExpPattern(self, pattern, captureCount=None):
    if captureCount is None:
        captureCount = getattr(self, '_legacyRegExpCaptureCount', 0)
    output = []
    i = 0
    length = len(pattern)
    classDepth = 0

    def sanitizeClassEscape(index):
        if index + 1 >= length:
            return '\\\\', index + 1

        escaped = pattern[index + 1]
        if escaped in '01234567':
            value, end = self.readLegacyRegExpOctalEscape(pattern, index)
            return '\\x%02x' % value, end

        if escaped == 'c':
            if index + 2 < length and pattern[index + 2] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz':
                value = ord(pattern[index + 2].upper()) - ord('A') + 1
                return '\\x%02x' % value, index + 3
            return 'c', index + 2

        if escaped == 'x':
            if index + 3 < length and Character.isHexDigit(pattern[index + 2]) and Character.isHexDigit(pattern[index + 3]):
                return pattern[index:index + 4], index + 4
            return 'x', index + 2

        if escaped == 'u':
            if index + 5 < length and all(Character.isHexDigit(ch) for ch in pattern[index + 2:index + 6]):
                return pattern[index:index + 6], index + 6
            return 'u', index + 2

        if escaped == 'b':
            return '\\x08', index + 2

        if escaped in 'fntvr0':
            return '\\' + escaped, index + 2

        if escaped in 'dDsSwW':
            return 'z' if index > 0 and pattern[index - 1] == '-' else 'a', index + 2

        if escaped in '^$\\.*+?()[]{}|-/':
            return '\\' + escaped if escaped in '\\]^-/' else re.escape(escaped), index + 2

        return re.escape(escaped), index + 2

    while i < length:
        ch = pattern[i]

        if classDepth > 0:
            if ch == '\\':
                replacement, i = sanitizeClassEscape(i)
                output.append(replacement)
                continue
            output.append(ch)
            if ch == '[':
                classDepth += 1
            elif ch == ']':
                classDepth -= 1
            i += 1
            continue

        if ch == '[':
            classDepth = 1
            output.append(ch)
            i += 1
            continue

        if ch == '\\':
            if i + 1 >= length:
                output.append(ch)
                i += 1
                continue

            escaped = pattern[i + 1]
            if escaped == 'c':
                if i + 2 < length and pattern[i + 2] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz':
                    value = ord(pattern[i + 2].upper()) - ord('A') + 1
                    output.append('\\x%02x' % value)
                    i += 3
                else:
                    output.append('c')
                    i += 2
                continue

            if escaped == 'x':
                if i + 3 < length and Character.isHexDigit(pattern[i + 2]) and Character.isHexDigit(pattern[i + 3]):
                    output.append(pattern[i:i + 4])
                    i += 4
                else:
                    output.append('x')
                    i += 2
                continue

            if escaped == 'u':
                if i + 5 < length and all(Character.isHexDigit(ch) for ch in pattern[i + 2:i + 6]):
                    output.append(pattern[i:i + 6])
                    i += 6
                else:
                    output.append('u')
                    i += 2
                continue

            if escaped in '123456789':
                end = i + 2
                while end < length and Character.isDecimalDigit(pattern[end]):
                    end += 1
                if int(pattern[i + 1:end]) <= captureCount:
                    output.append('(?:)')
                    i = end
                    continue

            if escaped in '01234567':
                value, end = self.readLegacyRegExpOctalEscape(pattern, i)
                output.append('\\x%02x' % value)
                i = end
                continue

            if escaped in '89':
                output.append(escaped)
                i += 2
                continue

            if escaped in 'pP':
                output.append(escaped)
                i += 2
                continue

            if escaped in 'bB':
                output.append('\\' + escaped)
                i += 2
                if i < length and pattern[i] == '{' and self.readRegExpBracedQuantifierSyntax(pattern, i) is None:
                    output.append('\\{')
                    i += 1
                continue

            if escaped in 'dDsSwWbBfnrtv0^$\\.*+?()[]{}|':
                output.append(pattern[i:i + 2])
            elif escaped == '/':
                output.append('/')
            else:
                output.append(re.escape(escaped))
            i += 2
            continue

        if ch == '{':
            end = self.readRegExpBracedQuantifierSyntax(pattern, i)
            if end is None:
                output.append('\\{')
                i += 1
            else:
                output.append(pattern[i:end])
                i = end
            continue

        output.append(ch)
        i += 1

    return ''.join(output)


_orig_testRegExp = Scanner.testRegExp

def testRegExp(self, pattern, flags):
    if 'u' in flags or 'v' in flags:
        return _orig_testRegExp(self, pattern, flags)
    previous = getattr(self, '_legacyRegExpCaptureCount', 0)
    try:
        self._legacyRegExpCaptureCount = self.countRegExpCapturingGroups(pattern)
    except Exception:
        self._legacyRegExpCaptureCount = 0
    try:
        return _orig_testRegExp(self, pattern, flags)
    finally:
        self._legacyRegExpCaptureCount = previous


Scanner.readLegacyRegExpOctalEscape = readLegacyRegExpOctalEscape
Scanner.sanitizeLegacyRegExpPattern = sanitizeLegacyRegExpPattern
Scanner.testRegExp = testRegExp
