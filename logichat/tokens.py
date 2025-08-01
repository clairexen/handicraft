import re, sys, types, numpy, itertools, collections
from dataclasses import dataclass, field
from en_basic import en_basic_words
import config, utils, tokens

# Still unused: ACK NAK SYN (and '\a', '\b', '\t', '\n', '\v', '\f', '\r')
ascii_ctrls = ["NUL", "SOH", "STX", "ETX", "EOT", "ENQ", "ACK", "BEL", "BS",
"HT", "LF", "VT", "FF", "CR", "SO", "SI", "DLE", "DC1", "DC2", "DC3", "DC4",
"NAK", "SYN", "ETB", "CAN", "EM", "SUB", "ESC", "FS", "GS", "RS", "US"]
ascii_ctrls_by_name = {n: i for i,n in enumerate(ascii_ctrls)}

example_text = """
REM "Tokenizer encode/decode example text."

REM ``
This is a 'Text'. 'i1' '110011' 'o3'
``

REM '''
This is an 'ASCII Text'. 'i1' '110011' 'o3'
'''

PROMPT "What is 7+4?"
TXT "11."

PROMPT "What's a short name for Robert?"
TXT "Bob."

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

TXT "Finished creating the table."

ENDMOD
"""


@dataclass
class Tokenizer:
    cfg: config._LogiChatConfig
    lines: list = field(default_factory=list)
    tokens: dict = field(default_factory=dict)
    encoder: dict = field(default_factory=dict)
    decoder: dict = field(default_factory=dict)

    def pr_table(self, /, showCtrl=False):
        lines = [l for l in self.lines if l is not None]

        if showCtrl:
            for i,l in enumerate(lines):
                idx = int(l.split(" ", 1)[0])
                if idx < len(ascii_ctrls):
                    lines[i] = f"{l[0:3]}{ascii_ctrls[idx]:<3}{l[3:]}"

        def render(cols):
            text = []
            col_height = (len(lines)+cols-1) // cols
            col_widths = [0]*cols

            for i in range(col_height):
                for j in range(cols):
                    k = i + col_height * j
                    l = lines[k] if k < len(lines) else ""
                    col_widths[j] = max(col_widths[j], len(l))

            for i in range(col_height):
                for j in range(cols):
                    k = i + col_height * j
                    l = lines[k] if k < len(lines) else ""
                    text.append(f"{l}\n" if j == cols-1 else f"{l:<{col_widths[j]}} | ")

            text.append(f"({len(lines)} tokens in total)\n")
            text = "".join(text)
            width = max(len(l) for l in text.split("\n"))
            return text, width

        import shutil
        term_size = utils.const(lambda: shutil.get_terminal_size())

        last_text = None
        for w in range(1, 20):
            text, width = render(w)
            if width > term_size.columns-3: break
            last_text = text
        print(last_text)

    def quote_str_char(self, c):
        if c == '"': return '"\\""'
        if c == '\n': return '"\\n"'
        if c == '\\': return '"\\\\"'
        if ord(c) < 31 or ord(c) >= 127:
            return f'"\\x{ord(c):02x}"'
        return f'"{c}"'

    def __post_init__(self):
        tok_shift = None
        tok_altgr = None

        def tok(x, tok_index=None):
            if tok_index is None:
                tok_index = len(self.lines)
                self.lines.append(None)
            else:
                if isinstance(tok_index, str):
                    tok_index = ascii_ctrls_by_name[tok_index]
                assert self.lines[tok_index] is None
            self.lines[tok_index] = f"{tok_index:<3} {x}"
            self.decoder[tok_index] = x
            self.tokens[x] = tok_index
            def enc(x, y):
                self.encoder[x] = y
            if x.startswith('"'):
                s = x.removeprefix('"')
                s = s.removesuffix('"')
                s = s.replace('" "', '')
                s = s.replace('\\"', '"')
                s = s.replace('\\n', '\n')
                s = s.replace('\\\\', '\\')
                s = tuple(self.quote_str_char(c) for c in s)
            else:
                s = (x,)
            if len(s) >= 1: enc(s[0], (tok_index,))
            if len(s) >= 2: enc(s[1], (tok_shift, tok_index))
            if len(s) >= 3: enc(s[2], (tok_altgr, tok_index))
            return tok_index

        # skip 7-bit ASCII range + 2 reserved slots for now
        while len(self.lines) < (128 + 2):
            self.lines.append(None)

        # only as EOF and ERROR marker
        tok("NULL", "NUL")

        # a break is just an empty line
        tok("BREAK", "ETB")

        # erase the last command
        tok("UNDO", 127)

        # chatbot interface:  PROMPT "<text>" ... REPLY "<text>"
        tok("PROMPT", "SOH")
        tok("REPLY", "EOT")

        # future extension: run a (python) query on SMT model
        tok("QUERY", "ENQ")

        # end of LLM generated output. e.g. after QUERY or REPLY
        tok("STOP", "CAN")

        # can be used anywhere
        tok("#", "DLE")    # #-comments at the end of a line
        tok("REM", "STX")  #   REM "This is just a remark that can be ignored"
        tok("TXT", "ETX")  #   TXT "This is relevant information and/or reply to a prompt or query"
        tok("TAG", "SUB")  #   TAG (i|f|n|o)<N> "This is an annotation of that entity"

        # start of module block
        tok("MODULE", 128) # MODULE ["<optional_name>"]

        # header statements
        tok("DIMS")        #   DIMS i<max> q<max> n<max> o<max> d<max> f<max> a<max>
        tok("NONE")        #   -
        tok("PI")          #   PI i<first> ... i<last>
        tok("PO")          #   PO o<first> ... o<last>
        tok("INIT")        #   INIT '1' q0 q1

        # (p)table blocks  #   [P]TABLE ["<optional_name>"]
        tok("TABLE")       #     GET i0 i1 i2
        tok("PTABLE")      #     GET i3 i4 i5 d1
        tok("GET")         #     SET o1 o2
        tok("SET")         #     SET q1
        tok("LUT_B", "FS") #     '010 1--0 ==> 1X 1'
        tok("LUT_D", "GS") #
        tok("LUT_T", "RS") #     '<3x[01-]> <4x[01-]> ==> <2x[01X-]> <[01X->'
        tok("LUT_E", "US") #     ^LUT_B    ^LUT_D      ^LUT_T       ^LUT_D  ^LUT_E

        # circuit block
        tok("CIRCUIT")     #   CIRCUIT ["<optional_name>"]
        tok("OPS")         #     OPS NAND NOR
        tok("DEF")         #     DEF n1 (NAND i1 i2)   # create/define a GATE
        tok("FUN_B","DC1") #     DEF n2 (MUX n1 i1 i3) # (MUX n1 i1 i3) = FUN_B MUX n1 i1 i3 FUN_E
        tok("FUN_E","DC2") #     DEF d1 n1             # also works with FF inputs (dN) and outputs (oN)

        # end of module block
        tok("ENDMOD", 129) # ENDMOD

        # OP types. (LUT<N> is directly followed by 3-state LUT data, terminated by LUT_E)
        for s in self.cfg.gates: tok(s)

        tok("STR_B", "DC3")  # normal "..."-strings: STR_B ... STR_E
        tok("STR_E", "DC4")  # here-doc-style strings: STR_B STR_B ... STR_E STR_E

        for idx in range(self.cfg.idx_base): tok(f"i{idx}")
        for idx in range(self.cfg.idx_base): tok(f"q{idx}")
        for idx in range(self.cfg.idx_base): tok(f"n{idx}")
        for idx in range(self.cfg.idx_base): tok(f"o{idx}")
        for idx in range(self.cfg.idx_base): tok(f"d{idx}")
        for idx in range(self.cfg.idx_base): tok(f"f{idx}")
        for idx in range(self.cfg.idx_base): tok(f"a{idx}")
        for idx in range(self.cfg.idx_base): tok(f"x{idx}")

        vals = set("01ZX")
        for n in range(1, self.cfg.max_nbits+1):
            for w in itertools.product(*["01ZX" for _ in range(n)]):
                vals.add("".join(w))
        for l,w in sorted((len(v),v) for v in vals):
            tok(f"'{w.replace('Z', '-')}'")

        if self.cfg.shift_altgr or self.cfg.with_words:
            tok_shift = tok("SHIFT", "SO")
        if self.cfg.shift_altgr:
            tok_altgr = tok("ALTGR", "SI")
        if self.cfg.with_words:
            tok("CAPS", "EM")

        def toks(ch, caps, alt):
            def myord(c):
                if len(c) == 1: return ord(c)
                if c == "\\n": return ord("\n")
                if c == "\\\\": return ord("\\")
                if c == "\\\"": return ord("\"")
                assert False
            if caps == ' ': caps = "\\n"
            if alt == '\\': alt = "\\\\"
            if alt == '\"': alt = "\\\""
            if self.cfg.shift_altgr:
                tok(f'"{ch}" "{caps}" "{alt}"', myord(ch))
            else:
                tok(f'"{ch}"', myord(ch))
                tok(f'"{caps}"', myord(caps))
                tok(f'"{alt}"', myord(alt))

        toks(*"aA!")
        toks(*"bB\"")
        toks(*"cC#")
        toks(*"dD$")
        toks(*"eE%")
        toks(*"fF&")
        toks(*"gG'")
        toks(*"hH(")
        toks(*"iI)")
        toks(*"jJ*")
        toks(*"kK+")
        toks(*"lL,")
        toks(*"mM-")
        toks(*"nN.")
        toks(*"oO/")
        toks(*"pP:")
        toks(*"qQ;")
        toks(*"rR<")
        toks(*"sS=")
        toks(*"tT>")
        toks(*"uU?")
        toks(*"vV@")
        toks(*"wW[")
        toks(*"xX\\")
        toks(*"yY]")
        toks(*"zZ^")
        toks(*"05`")
        toks(*"16{")
        toks(*"27|")
        toks(*"38}")
        toks(*"49~")
        toks(*"  _")

        morphemes = []
        if self.cfg.with_words:
            morphemes += [f"_{w}" for w in en_basic_words]

        if self.cfg.with_dbls:
            dbls = set()
            for w in en_basic_words:
                if len(w) <= 2: continue
                for i in range(0,len(w)-1):
                    dbls.add(w[i:i+2])
            for dbl in dbls:
                morphemes.append(f".{dbl}")

        if self.cfg.with_tris:
            tris = set()
            for w in en_basic_words:
                if len(w) <= 3: continue
                for i in range(0,len(w)-2):
                    tris.add(w[i:i+3])
            for tri in tris:
                morphemes.append(f".{tri}")

        if morphemes:
            for _,_,t in sorted((len(t), t[1:]+t[0], t) for t in morphemes):
                tok(t)

        opnames = " | ".join(t for t in self.cfg.gates)
        objnames = " | ".join(t for t in self.decoder.values() if t[0] in 'iqnodfa')
        self.re_keywords = re.compile(f"""
            (?<![a-zA-Z0-9]) ( PROMPT | REPLY | QUERY | REM | TXT | TAG | MODULE | DIMS | PI | PO |
                    TABLE | PTABLE | GET | SET | CIRCUIT | OPS | DEF | ENDMOD | {opnames} | {objnames}) (?![a-zA-Z0-9])
        """, re.A|re.X)

        vocab_size = max(self.decoder.keys())+1
        stoi = { f" {s}": i for i, s in self.decoder.items() }
        itos = { i: f" {s}" for i, s in self.decoder.items() }

        if False:
            for i in range(vocab_size):
                if i not in itos:
                    itos[i] = f" *UNUSED_TOKEN_{i}*"
                    stoi[f" *UNUSED_TOKEN_{i}*"] = i

        # nanoGPT meta.pkl
        self.meta = {
            'vocab_size': vocab_size,
            'itos': itos,
            'stoi': stoi,
        }

        self.binext = "uint8" if vocab_size < 256 else "uint16"
        self.bintype = numpy.uint8 if vocab_size < 256 else numpy.uint16

    def encode(self, text, encodeText=True):
        tokens = []
        state_str1 = False
        state_str2 = False
        state_str3 = False
        state_bits = False
        pos = 0

        while pos < len(text):
            if state_str1 or state_str2 or state_str3:
                if not encodeText:
                    l = 0; t = []
                    if state_str1:
                        while text[pos+l] != '"' and pos+l < len(text):
                            l += 1 if text[pos+l] != "\\" else 2
                        t += self.encoder["STR_E"]
                    elif state_str2:
                        while text[pos+l:pos+l+4] != "\n``\n" and pos+l < len(text):
                            l += 1
                        t += self.encoder["STR_E"]
                        t += self.encoder["STR_E"]
                    else:
                        assert state_str3
                        while text[pos+l:pos+l+5] != "\n'''\n" and pos+l < len(text):
                            l += 1
                        t += self.encoder["STR_E"]
                        t += self.encoder["STR_E"]
                        t += self.encoder["STR_E"]
                    tokens += [text[pos:pos+l]] + t
                    pos += l + (1 if state_str1 else 4)
                    state_str1 = False; state_str2 = False; state_str3 = False
                    continue

                elif state_str1:
                    if text[pos] == '"':
                        state_str1 = False; t = 'STR_E'
                    elif text[pos:pos+2] == '\\n':
                        pos += 1; t ='"\\n"'
                    elif text[pos:pos+2] in ('\\"', '\\\\'):
                        pos += 1; t ='"\\{text[pos]}"'
                    else:
                        t = self.quote_str_char(text[pos])
                    assert t in self.encoder, f"Token {t} not in encoder table."
                    tokens += self.encoder[t]
                    pos += 1
                    continue

                elif state_str2:
                    if text[pos:].startswith('\n``\n'):
                        state_str2 = False; t = 'STR_E'
                        tokens += self.encoder[t]
                        pos += 3
                    else:
                        t = self.quote_str_char(text[pos])
                    assert t in self.encoder, f"Token {t} not in encoder table."
                    tokens += self.encoder[t]
                    pos += 1
                    continue

                else:
                    assert state_str3
                    if text[pos:].startswith("\n'''\n"):
                        state_str2 = False; t = 'STR_E'
                        tokens += self.encoder[t]
                        pos += 3
                    else:
                        t = self.quote_str_char(text[pos])
                    assert t in self.encoder, f"Token {t} not in encoder table."
                    tokens += self.encoder[t]
                    pos += 1
                    continue

            if state_bits:
                if text[pos] == "'":
                    state_bits = False; pos += 1
                    tokens += self.encoder["LUT_E"]
                    continue
                if text[pos:].startswith(" ==> "):
                    t = "LUT_T"; pos += 4
                elif text[pos] == " ":
                    t = "LUT_D"
                elif text[pos] in "01-X":
                    t = f"'{text[pos]}'"
                else:
                    tokens.append(0)
                    break
                pos += 1
                tokens += self.encoder[t]
                continue

            # no special state
            if text[pos:pos+2] == "\n\n":
                pos += 1
                tokens += self.encoder["BREAK"]
                continue
            if text[pos] in " \t\n":
                pos += 1
                continue
            if m := self.re_keywords.match(text, pos):
                pos += len(m[0])
                assert m[0] in self.encoder
                tokens += self.encoder[m[0]]
                continue
            if text[pos] == '"':
                pos += 1
                state_str1 = True
                tokens += self.encoder['STR_B']
                continue
            if text[pos:].startswith("``\n"):
                pos += 3
                state_str2 = True
                tokens += self.encoder['STR_B']
                tokens += self.encoder['STR_B']
                continue
            if text[pos:].startswith("'''\n"):
                pos += 4
                state_str3 = True
                tokens += self.encoder['STR_B']
                tokens += self.encoder['STR_B']
                tokens += self.encoder['STR_B']
                continue
            if text[pos] == "'":
                pos += 1
                state_bits = True
                tokens += self.encoder['LUT_B']
                continue

            if text[pos] == "(":
                pos += 1
                tokens += self.encoder['FUN_B']
                continue

            if text[pos] == ")":
                pos += 1
                tokens += self.encoder['FUN_E']
                continue

            tokens.append(0)
            tokens.append(text[pos:pos+15])
            break

        return tokens

    def decode(self, tokens):
        text = []
        pos = 0

        def last_c():
            while text and not text[-1]:
                text.pop()
            if not text:
                return None
            return text[-1][-1]

        tok = None
        while pos < len(tokens):
            if isinstance(tokens[pos], str):
                text.append(tokens[pos])
                pos += 1
                continue

            last_tok = tok
            tok = self.decoder[tokens[pos]]
            pos += 1

            if tok in (*("REM TXT PROMPT QUERY MODULE ENDMOD".split()),
                       *(indent_2 := "DIMS TABLE PTABLE CIRCUIT".split()),
                       *(indent_4 := "PI PO GET SET OPS DEF".split())):
                if last_c() is not None:
                    text.append('\n')
                    if tok in indent_2:
                        text.append('  ')
                    if tok in indent_4:
                        text.append('    ')
                text.append(tok)
                continue

            if tok == 'BREAK':
                text.append('\n')
                continue

            if tok[0] in "iqnodfa" or tok in self.cfg.gates:
                if last_tok != "FUN_B":
                    text.append(f' {tok}')
                else:
                    text.append(tok)
                continue

            if tok == 'STR_B':
                state_str = True
                text.append(' ')
                if tokens[pos-1] == tokens[pos]:
                    if pos+1 < len(tokens) and tokens[pos-1] == tokens[pos+1]:
                        text.append("'''\n")
                        pos += 2
                    else:
                        text.append('``\n')
                        pos += 1
                else:
                    text.append('"')
                continue

            if tok == 'STR_E':
                state_str = False
                if tokens[pos-1] == tokens[pos]:
                    if pos+1 < len(tokens) and tokens[pos-1] == tokens[pos+1]:
                        text.append("\n'''")
                        pos += 2
                    else:
                        text.append('\n``')
                        pos += 1
                else:
                    text.append('"')
                continue

            if tok in ('LUT_B', 'LUT_E'):
                state_lut = tok == 'LUT_B'
                text.append(" '" if state_lut else "'")
                continue

            if tok == 'LUT_D':
                text.append(' ')
                continue

            if tok == 'LUT_T':
                text.append(' ==> ')
                continue

            if tok == 'FUN_B':
                text.append(' (')
                continue

            if tok == 'FUN_E':
                text.append(')')
                continue

            if tok[0] == "'":
                text.append(tok.strip("'") if state_lut else tok)
                continue

            if tok[0] == '"':
                text.append(tok.removeprefix('"').removesuffix('"'))
                continue

            pos -= 1
            text.append(f"*** DECODE ERROR AT POSITION {pos}: {tok} ***")
            break

        return "".join(text)

    def tok2str(self, toks):
        if isinstance(toks, str):
            return repr(toks)
        if isinstance(toks, int):
            s = self.decoder[toks]
            if ' ' in s and s != '" "':
                s = s.removeprefix('\"')
                s = s.removesuffix('\"')
                s = s.replace('" "', '')
                s = s.replace('\\"', '"')
                s = f"[{s}]"
            return s
        if toks is None:
            return "None"
        return " ".join(self.tok2str(t) for t in toks)

def main():
    if "-t" in opts or "-T" in opts:
        if not args:
            args.append(example_text)
        cfg = config.cfg_large
        lex = cfg.lex()
        lex.pr_table(8)
        for s in args:
            print()
            print(f"Input: {s}")
            t = lex.encode(s, "-T" in opts)
            print(f"Ids: {' '.join(repr(i) for i in t)}")
            print(f"Tokens: {lex.tok2str(t)}")
            x = lex.decode(t)
            print(f"Output: {x}")
        return 0

    for c, t in config.cfgs.values():
        print("\n" + (t := f"{t} Token List"))
        print("=" * len(t))
        c.lex().pr_table(showCtrl=("-c" in opts))

if __name__ == "__main__":
    cmdname, *args = sys.argv
    opts = set(a for a in args if a.startswith("-"))
    args = [a for a in args if a not in opts]
    sys.exit(main())
