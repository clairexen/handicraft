import pcre2, sys, types, numpy, itertools, collections
from dataclasses import dataclass, field
from en_basic import en_basic_words
import config, utils, tokens

# Still unused: ACK NAK SYN (and '\a', '\b', '\t', '\n', '\v', '\f', '\r')
ascii_ctrls = ["NUL", "SOH", "STX", "ETX", "EOT", "ENQ", "ACK", "BEL", "BS",
"HT", "LF", "VT", "FF", "CR", "SO", "SI", "DLE", "DC1", "DC2", "DC3", "DC4",
"NAK", "SYN", "ETB", "CAN", "EM", "SUB", "ESC", "FS", "GS", "RS", "US"]
ascii_ctrls_by_name = {n: i for i,n in enumerate(ascii_ctrls)}

cls_ws_etc = set(" \a\b\t\n\x1b")
cls_abc = set("abcdefghijklmnopqrstuvwxyz")
cls_ABC = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
cls_123 = set("1234567890")
cls_abcABC123 = cls_abc | cls_ABC | cls_123
cls_special = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")

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

    base_embd_map: dict = field(default_factory=dict)
    base_embd_names: list = field(default_factory=list)

    modif_embd_map: dict = field(default_factory=dict)
    modif_embd_names: list = field(default_factory=list)

    token_map: dict = field(default_factory=dict)
    token_names: list = field(default_factory=list)

    kwtoi_map: dict = field(default_factory=dict)
    stoi_map: dict = field(default_factory=dict)
    itos_map: dict = field(default_factory=dict)

    def pr_table(self, /, showCtrl=False):
        lines = [f"{i:3} {'----' if n is None else n}" for i,n in enumerate(self.token_names)]

        if showCtrl:
            for i,l in enumerate(lines):
                if i >= len(ascii_ctrls): break
                lines[i] = f"{l[0:3]} {ascii_ctrls[i]:<3}{l[3:]}"

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

            text.append(f"({len(self.token_names)} tokens in total, " +
                        f"{len(self.base_embd_names)} base embeddings, " +
                        f"{len(self.modif_embd_names)} modifier embeddings)\n")
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

    def add_base_embd(self, base_name):
        assert base_name not in self.base_embd_map
        self.base_embd_map[base_name] = (idx := len(self.base_embd_names))
        self.base_embd_names.append(base_name)
        return idx

    def add_modif_embd(self, modif_name):
        assert modif_name not in self.modif_embd_map
        self.modif_embd_map[modif_name] = (idx := len(self.modif_embd_names))
        self.modif_embd_names.append(modif_name)
        return idx

    def add_token(self, token_name, base_idx=None, modif_idx=None, idx=None):
        if base_idx and modif_idx is None and idx is None:
            idx = base_idx; base_idx = None

        if kwmode := (not base_idx and not modif_idx):
            base_idx = self.add_base_embd(f"B_{token_name}")
            modif_idx = 0

        assert token_name not in self.token_map

        if idx is None:
            idx = len(self.token_names)
            self.token_names.append(None)
        else:
            if isinstance(idx, str):
                idx = ascii_ctrls.index(idx)
        assert self.token_names[idx] is None

        if kwmode:
            self.add_kwtoi(token_name, idx)

        if isinstance(base_idx, str): base_idx = self.base_embd_map[base_idx]
        if isinstance(modif_idx, str): modif_idx = self.modif_embd_map[modif_idx]

        self.token_map[token_name] = (idx, base_idx, modif_idx)
        self.token_names[idx] = token_name
        return idx

    def add_stoi(self, s, i):
        self.stoi_map[s] = i
        self.itos_map[i] = s

    def add_kwtoi(self, s, i):
        self.kwtoi_map[s] = i
        self.itos_map[i] = s

    def __post_init__(self):
        self.add_modif_embd("M_NOMOD")
        self.add_modif_embd("M_SHIFT")
        self.add_modif_embd("M_CAPS")
        self.add_modif_embd("M_SPACE")
        self.add_modif_embd("M_SPACE_SHIFT")
        self.add_modif_embd("M_SPACE_CAPS")
        for i in range(self.cfg.idx_base):
            self.add_modif_embd(f"M_{i}")

        # skip 7-bit ASCII range + 2 reserved slots for now
        while len(self.token_names) < 128:
            self.token_names.append(None)

        # only as EOF and ERROR marker
        self.add_token("NULL", 0, 0, "NUL")

        # a break is just an empty line
        self.add_token("BREAK", "CR")

        # erase the last command
        self.add_token("UNDO", 127)

        # chatbot interface:  PROMPT "<text>" ... REPLY "<text>"
        self.add_token("PROMPT", "SOH")
        self.add_token("REPLY", "EOT")

        # future extension: run a (python) query on SMT model
        self.add_token("QUERY", "ENQ")

        # end of LLM generated output. e.g. after QUERY or REPLY
        self.add_token("STOP", "CAN")

        # can be used anywhere
        self.add_token("#", "DLE")    # #-comments at the end of a line
        self.add_token("REM", "STX")  #   REM "This is just a remark that can be ignored"
        self.add_token("TXT", "ETX")  #   TXT "This is relevant information and/or reply to a prompt or query"
        self.add_token("TAG", "SUB")  #   TAG (i|f|n|o)<N> "This is an annotation of that entity"

        # start of module block
        self.add_token("MODULE", "ACK") # MODULE ["<optional_name>"]

        # header statements
        self.add_token("DIMS")        #   DIMS i<max> q<max> n<max> o<max> d<max> f<max> a<max>
        self.add_token("NONE", "EM")  #   -
        self.add_token("PI")          #   PI i<first> ... i<last>
        self.add_token("PO")          #   PO o<first> ... o<last>
        self.add_token("INIT")        #   INIT '1' q0 q1

        # (p)table blocks  #   [P]TABLE ["<optional_name>"]
        self.add_token("TABLE")       #     GET i0 i1 i2
        self.add_token("PTABLE")      #     GET i3 i4 i5 d1
        self.add_token("GET", "SYN")  #     SET o1 o2
        self.add_token("SET", "ETB")  #     SET q1
        self.add_token("LUT_B", "FS") #     '010 1--0 ==> 1X 1'
        self.add_token("LUT_D", "GS") #
        self.add_token("LUT_T", "RS") #     '<3x[01-]> <4x[01-]> ==> <2x[01X-]> <[01X->'
        self.add_token("LUT_E", "US") #     ^LUT_B    ^LUT_D      ^LUT_T       ^LUT_D  ^LUT_E

        # circuit block
        self.add_token("CIRCUIT")     #   CIRCUIT ["<optional_name>"]
        self.add_token("OPS")         #     OPS NAND NOR
        self.add_token("DEF")         #     DEF n1 (NAND i1 i2)   # create/define a GATE
        self.add_token("FUN_B", "SO") #     DEF n2 (MUX n1 i1 i3) # (MUX n1 i1 i3) = FUN_B MUX n1 i1 i3 FUN_E
        self.add_token("FUN_E", "SI") #     DEF d1 n1             # also works with FF inputs (dN) and outputs (oN)

        # end of module block
        self.add_token("ENDMOD", "NAK") # ENDMOD

        # OP types. (LUT<N> is directly followed by 3-state LUT data, terminated by LUT_E)
        for s in self.cfg.gates: self.add_token(s)

        self.add_token("STR_B", "VT")  # normal "..."-strings: STR_B ... STR_E
        self.add_token("STR_E", "FF")  # here-doc-style strings: STR_B STR_B ... STR_E STR_E

        for kind in "iqnodfax":
            base_idx = self.add_base_embd(f"B_{kind}")
            for idx in range(min(self.cfg.idx_base, 10)):
                self.add_kwtoi(f"{kind}{idx}", self.add_token(f"{kind}{idx}", base_idx, f"M_{idx}"))

        for i,(w,n) in enumerate(zip("01ZX", "DC1 DC2 DC3 DC4".split())):
            self.add_token(f"'{w.replace('Z', '-')}'", n)

        vals = set()
        for n in range(2, 1 + min(2, self.cfg.max_nbits)):
            for w in itertools.product(*["01ZX" for _ in range(n)]):
                vals.add("".join(w))
        for l,w in sorted((len(v),v) for v in vals):
            self.add_token(f"'{w.replace('Z', '-')}'")

        while len(self.token_names) < 256:
            self.token_names.append(None)

        for ch in sorted(cls_ws_etc):
            t = f'"{repr(ch)[1:-1]}"'
            base_idx = self.add_base_embd(f"B:{t}")
            self.add_stoi(ch, self.add_token(t, base_idx, 0, ord(ch)))

        for ch in sorted(cls_special):
            if (c := ch) in '"\\': ch = "\\" + ch
            base_idx = self.add_base_embd("B:" + (t := f'"{ch}"'))
            self.add_stoi(c, self.add_token(t, base_idx, 0, ord(c)))

        for ch in sorted(cls_123):
            base_idx = self.add_base_embd(f"B:{ch}")
            self.add_stoi(      ch,         self.add_token("." + ch,         base_idx, "M_NOMOD",       ord(ch)))
            if self.cfg.with_words:
                self.add_stoi(" " + ch,         self.add_token("_" + ch,         base_idx, "M_SPACE",       None))

        for ch in sorted(cls_abc):
            base_idx = self.add_base_embd(f"B:{ch}")
            self.add_stoi(      ch,         self.add_token("." + ch,         base_idx, "M_NOMOD",       ord(ch)))
            self.add_stoi(      ch.upper(), self.add_token("." + ch.upper(), base_idx, "M_SHIFT",       ord(ch.upper())))
            if self.cfg.with_words:
                self.add_stoi(" " + ch,         self.add_token("_" + ch,         base_idx, "M_SPACE",       None))
                self.add_stoi(" " + ch.upper(), self.add_token("_" + ch.upper(), base_idx, "M_SPACE_SHIFT", None))

        vals = set()
        for n in range(3, 1 + self.cfg.max_nbits):
            for w in itertools.product(*["01ZX" for _ in range(n)]):
                vals.add("".join(w))
        for l,w in sorted((len(v),v) for v in vals):
            self.add_token(f"'{w.replace('Z', '-')}'")

        morphemes = set()
        if self.cfg.with_words:
            morphemes |= set(w for w in en_basic_words if len(w) > 1)
            for k in [2,3]:
                for w in en_basic_words:
                    if len(w) <= k: continue
                    for i in range(0,len(w)-k+1):
                        morphemes.add(w[i:i+k])

        for _,t in sorted((len(t),t) for t in morphemes):
            if len(t) < 2: continue
            t_caps = t.upper(); t_shift = t_caps[0] + t[1:]
            base_idx = self.add_base_embd(f"B:{t}")
            self.add_stoi(      t,       self.add_token("." + t,       base_idx, "M_NOMOD"))
            self.add_stoi(      t_shift, self.add_token("." + t_shift, base_idx, "M_SHIFT"))
            self.add_stoi(      t_caps,  self.add_token("." + t_caps,  base_idx, "M_CAPS"))
            self.add_stoi(" " + t,       self.add_token("_" + t,       base_idx, "M_SPACE"))
            self.add_stoi(" " + t_shift, self.add_token("_" + t_shift, base_idx, "M_SPACE_SHIFT"))
            self.add_stoi(" " + t_caps,  self.add_token("_" + t_caps,  base_idx, "M_SPACE_CAPS"))

        for idx in range(10, self.cfg.idx_base):
            for kind in "iqnodfax":
                self.add_token(f"{kind}{idx}", f"B_{kind}", f"M_{idx}")

        opnames = " | ".join(t for t in self.cfg.gates)
        objnames = " | ".join(t for t in self.token_names if t and t[0] in 'iqnodfa')
        self.re_keywords = pcre2.compile(f"""
            (?<![a-zA-Z0-9]) ( PROMPT | REPLY | QUERY | REM | TXT | TAG | MODULE | DIMS | PI | PO |
                    TABLE | PTABLE | GET | SET | CIRCUIT | OPS | DEF | ENDMOD | {opnames} | {objnames}) (?![a-zA-Z0-9])
        """, pcre2.X)

        if self.cfg.with_words:
            sorted_by_len = lambda l: [t for _,_,t in sorted((-len(t), t.lower(), t) for t in l)]
            morph_pattern = sorted_by_len(f"[{w[0]}{w[0].upper()}]{w[1:]}|{w.upper()}" for w in morphemes)
            morph_pattern = f" ?(?:{'|'.join(morph_pattern)}|[a-zA-Z0-9])"
            self.re_morph = pcre2.compile(morph_pattern)
        else:
            self.re_morph = None

        # nanoGPT meta.pkl
        self.meta = {
            'vocab_size': len(self.base_embd_names),
            'modif_size': len(self.modif_embd_names),
            'tokens': [None if self.token_names[i] is None else \
                            (self.token_names[i], *self.token_map[self.token_names[i]][1:])
                                    for i in range(len(self.token_names))],
            'kwtoi': self.kwtoi_map,
            'stoi': self.stoi_map,
            'itos': self.itos_map,
        }

        self.binext = "uint8" if len(self.token_names) < 256 else "uint16"
        self.bintype = numpy.uint8 if len(self.token_names) < 256 else numpy.uint16

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
                        t += [self.kwtoi_map["STR_E"]]
                    elif state_str2:
                        while text[pos+l:pos+l+4] != "\n``\n" and pos+l < len(text):
                            l += 1
                        t += [self.kwtoi_map["STR_E"]]*2
                    else:
                        assert state_str3
                        while text[pos+l:pos+l+5] != "\n'''\n" and pos+l < len(text):
                            l += 1
                        t += [self.kwtoi_map["STR_E"]]*3
                    tokens += [text[pos:pos+l]] + t
                    pos += l + (1 if state_str1 else 4)
                    state_str1 = False; state_str2 = False; state_str3 = False
                    continue

                if self.re_morph and (m := self.re_morph.match(text, pos)):
                    tokens += [self.stoi_map[m[0]]]
                    pos += len(m[0])
                    continue

                if state_str1:
                    if text[pos] == '"':
                        pos += 1
                        state_str1 = False
                        tokens += [self.kwtoi_map['STR_E']]
                    else:
                        if text[pos:pos+2] == '\\n':
                            pos += 1; t = self.token_map['"\\n"'][0]
                        elif text[pos:pos+2] in ('\\"', '\\\\'):
                            pos += 1; t = self.token_map[f'"\\{text[pos]}"'][0]
                        else:
                            t = self.stoi_map[text[pos]]
                        tokens += [t]; pos += 1
                    continue

                elif state_str2:
                    if text[pos:].startswith('\n``\n'):
                        pos += 3
                        state_str2 = False
                        tokens += [self.kwtoi_map['STR_E']]*2
                    else:
                        t = self.stoi_map[text[pos]]
                        tokens += [t]; pos += 1
                    continue

                else:
                    assert state_str3
                    if text[pos:].startswith("\n'''\n"):
                        pos += 4
                        state_str3 = False
                        tokens += [self.kwtoi_map['STR_E']]*3
                    else:
                        t = self.stoi_map[text[pos]]
                        tokens += [t]; pos += 1
                    continue

            if state_bits:
                if text[pos] == "'":
                    state_bits = False; pos += 1
                    tokens += [self.kwtoi_map["LUT_E"]]
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
                tokens += [self.kwtoi_map[t]]
                continue

            # no special state
            if text[pos:pos+2] == "\n\n":
                pos += 1
                tokens += [self.kwtoi_map["BREAK"]]
                continue
            if text[pos] in " \t\n":
                pos += 1
                continue
            if m := self.re_keywords.match(text, pos):
                pos += len(m[0])
                assert m[0] in self.kwtoi_map
                tokens += [self.kwtoi_map[m[0]]]
                continue
            if text[pos] == '"':
                pos += 1
                state_str1 = True
                tokens += [self.kwtoi_map['STR_B']]
                continue
            if text[pos:].startswith("``\n"):
                pos += 3
                state_str2 = True
                tokens += [self.kwtoi_map['STR_B']]*2
                continue
            if text[pos:].startswith("'''\n"):
                pos += 4
                state_str3 = True
                tokens += [self.kwtoi_map['STR_B']]*3
                continue
            if text[pos] == "'":
                pos += 1
                state_bits = True
                tokens += [self.kwtoi_map['LUT_B']]
                continue

            if text[pos] == "(":
                pos += 1
                tokens += [self.kwtoi_map['FUN_B']]
                continue

            if text[pos] == ")":
                pos += 1
                tokens += [self.kwtoi_map['FUN_E']]
                continue

            if text[pos] == "\x00":
                pos += 1
                tokens += [self.kwtoi_map['NULL']]
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
                tok = None
                pos += 1
                continue

            last_tok = tok
            tokidx = tokens[pos]
            tok = self.token_names[tokidx]
            pos += 1

            if tok[0] in "._\"":
                text.append(self.itos_map[tokidx])
                continue

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
                    if pos-1: text.append(' ')
                text.append(tok)
                continue

            if tok == 'STR_B':
                state_str = True
                if pos-1: text.append(' ')
                if pos < len(tokens) and tokens[pos-1] == tokens[pos]:
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
                if pos < len(tokens) and tokens[pos-1] == tokens[pos]:
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
                if state_lut and pos-1: text.append(' ')
                text.append("'")
                continue

            if tok == 'LUT_D':
                text.append(' ')
                continue

            if tok == 'LUT_T':
                text.append(' ==> ')
                continue

            if tok == 'FUN_B':
                if pos-1: text.append(' ')
                text.append('(')
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
            s = self.token_names[toks]
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

        cfg = config.cfg
        for n,(c,_) in config.cfgs.items():
            if f"-{n}" in opts: cfg = c
        lex = cfg.lex()
        lex.pr_table()

        for s in args:
            print()
            print(f"Input: {s}")
            t = lex.encode(s, "-T" in opts)
            print(f"Ids: {' '.join(repr(i) for i in t)}")
            print(f"Tokens: {lex.tok2str(t)}")
            x = lex.decode(t)
            print(f"Output: {x}")

        return 0

    if "-e" in opts:
        cfg = config.cfg
        for n,(c,_) in config.cfgs.items():
            if f"-{n}" in opts: cfg = c
        lex = cfg.lex()

        with open(args[0], "ab" if "-a" in opts else "wb") as f:
            data = sys.stdin.read()
            data = data.split("\x00")
            for t in data:
                if not t: continue
                t = lex.encode(t) + [0]
                t = numpy.array(t, lex.bintype)
                t.tofile(f)
                f.flush()

        return 0

    for c, t in config.cfgs.values():
        print("\n" + (t := f"{t} Token List"))
        print("=" * len(t))
        c.lex().pr_table(showCtrl=("-c" in opts))

if __name__ == "__main__":
    opts, args = utils.opts_args()
    cmdname, *args = sys.argv
    opts = set(a for a in args if a.startswith("-"))
    args = [a for a in args if a not in opts]
    sys.exit(main())
