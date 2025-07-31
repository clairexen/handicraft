import re
from dataclasses import dataclass, field

@dataclass
class Stmt:
    src : str
    childidx : int
    nchildren : int = None
    tokens: list = field(default_factory=list)

    def unparse(self, showTokens=False):
        if self.tokens[0] == "BREAK":
            return ""
        indent = ""
        if self.tokens[0] in ('DIMS', 'PI', 'PO', 'TABLE', 'PTABLE', 'CIRCUIT'):
            indent = "  "
        if self.tokens[0] in ('GET', 'SET', 'DEF', 'OPS'):
            indent = "    "
        if showTokens:
            return f"{indent}<{'> <'.join(self.tokens)}>"
        return f"{indent}{' '.join(self.tokens)}"

@dataclass
class Code:
    lines: list = field(default_factory=list)
    stmts: list = field(default_factory=list)
    names: dict = field(default_factory=dict)

    def unparse(self, showTokens=False):
        return "\n".join(s.unparse(showTokens) for s in self.stmts)

class Parser:
    def __init__(self, text = None):
        self.code = Code()
        self.mod = None

        if text is not None:
            self.parse(text)

    def scan(self, lines, pos):
        text = lines[pos]
        self.code.lines.append(lines[pos])
        pos += 1

        if text.endswith(" ``"):
            while not text.endswith("\n``"):
                text += "\n" + lines[pos]
                self.code.lines.append(lines[pos])
                pos += 1

        elif text.endswith(" '''"):
            while not text.endswith("\n'''"):
                text += "\n" + lines[pos]
                self.code.lines.append(lines[pos])
                pos += 1

        re_tok = re.compile(r"""
            [ \t\r\n]+ | [A-Z]+ | [iqnodfa][0-7]+ |
            "( [^"\\] | \\. )+" | (``|''')\n.* |
            '( [-01X ] | [ ]==>[ ] )+' | [()]
        """, re.S|re.X)

        p = 0
        toks = []
        while p < len(text):
            if m := re_tok.match(text, p):
                if m[0][0] not in " \t\r\n":
                    toks.append(m[0])
                p += len(m[0])
                continue
            assert False, f"Scanner Error at {repr(text[p:p+10])}\nIn: {repr(text)}"

        if not toks:
            toks.append("BREAK")

        return toks, text, pos

    def parse(self, text):
        c = self.code
        lines = text.strip().split("\n")
        pos = 0

        while pos < len(lines):
            linenr = len(self.code.lines)+1
            toks, txt, pos = self.scan(lines, pos)
            s = Stmt(txt, len(c.stmts))
            s.tokens = toks
            idx = len(c.stmts)
            c.stmts.append(s)

            if toks[0] == "MODULE":
                if len(toks) > 1:
                    name = toks[1].removeprefix('"').removesuffix('"')
                else:
                    name = f"_{toks[0]}_L{linenr}"
                c.names[name] = {"": (idx, linenr)}
                self.mod = name

            if toks[0] == "ENDMOD":
                self.mod = None

            if toks[0] in ('TABLE', 'PTABLE', 'CIRCUIT') and self.mod:
                if len(toks) > 1:
                    name = toks[1].removeprefix('"').removesuffix('"')
                else:
                    name = f"_{toks[0]}_L{linenr}"
                c.names[self.mod][name] = (idx, linenr)

example_text = """
REM "Tokenizer encode/decode example text."

REM ``
This is a 'Text'. 'i1' '110011' 'o3'
``

REM '''
This is an 'ASCII Text'. 'i1' '110011' 'o3'
'''

PROMPT "What is 7+4?"
REPLY "11."

PROMPT "What's a short name for Robert?"
REPLY "Bob."

MODULE "test_1"
  DIMS i7 q7 n7 o7 d7 f7 a7
  PI i1 i2 i3 i4
  PO o1 o2

  PTABLE "ref"
    GET i1 i2
    GET i3 i4
    SET o1 o2
    DEF '11 -- ==> 1-'
    DEF '-- 00 ==> -0'
    DEF '-- -- ==> 01'

  CIRCUIT "impl"
    DEF o1 (AND i1 i2)
    DEF o2 (OR i3 i4)

PROMPT ``
Create another truth table, that explicitly encodes the
cases where the AND-inputs are '1' and the OR-inputs are '0'.
``
REM ``
Thinking.. The user asks me to ... the PTABLE "ref" has one line
per gate, plus a final line with defaults ... in order to get
the encoding the user asks for I therefore should ...
``

  PTABLE "alt"
    GET i1 i2
    GET i3 i4
    SET o1 o2
    DEF '0- -- ==> 0-'
    DEF '-0 -- ==> 0-'
    DEF '-- 1- ==> -1'
    DEF '-- -1 ==> -1'
    DEF '-- -- ==> 10'

REPLY "Finished creating the table."

ENDMOD
"""

if __name__ == "__main__":
    p = Parser(example_text)
    print(p.code.unparse(True))
    print(p.code.names)
