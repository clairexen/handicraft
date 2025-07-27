import types
from itertools import product

def reload():
    exec(open("tokens.py").read(), globals())

def pr(x, *, keys=None):
    if isinstance(x, types.GeneratorType):
        x = list(x)

    if isinstance(keys, str):
        keys = keys.split()
    elif keys is None:
        if isinstance(x, str):
            print(repr(x))
            return
        if isinstance(x, (tuple, list, dict, set, frozenset)):
            if len(s := repr(x)) < 100:
                print(s)
                return
            print(s)
            return

    print(str(x))
    def val2str(x):
        if isinstance(x, (str, tuple, list, dict, set, frozenset)):
            return repr(x)
        if isinstance(x, (bytes)):
            return tuple(x)
        return x
    print("  `- " + "\n  `- ".join(f"{k:<20} {val2str(getattr(x,k))}" for k in (keys if keys else dir(x))))

# =======================================================

from dataclasses import dataclass, field

@dataclass
class GraphSprechConfig:
    num_pi: int = 16
    num_ff: int = 8
    num_op: int = 64
    num_po: int = 4
    nbits_3state: int = 2
    nbits_4state: int = 2
    shift_altgr: bool = False

@dataclass
class TokenList:
    lines: list = field(default_factory=list)
    encoder: dict = field(default_factory=dict)

    def pr_table(self, cols=7, /):
        col_height = (len(tokens.lines)+cols-1) // cols
        col_widths = [0]*cols

        for i in range(col_height):
            for j in range(cols):
                k = i + col_height * j
                l = tokens.lines[k] if k < len(tokens.lines) else ""
                col_widths[j] = max(col_widths[j], len(l))

        for i in range(col_height):
            for j in range(cols):
                k = i + col_height * j
                l = tokens.lines[k] if k < len(tokens.lines) else ""
                print(f"{l}\n" if j == cols-1 else f"{l:<{col_widths[j]}} | ", end="")

def gentokens(cfg: GraphSprechConfig = GraphSprechConfig()):
    ret = TokenList()
    tok_shift = None
    tok_altgr = None

    def tok(x):
        s = x.split()
        tok_index = len(ret.lines)
        ret.lines.append(f"{tok_index:<3} {x}")
        def enc(x, y): ret.encoder[x] = y
        if len(s) >= 1: enc(s[0], (tok_index,))
        if len(s) >= 2: enc(s[1], (tok_shift, tok_index))
        if len(s) >= 3: enc(s[2], (tok_altgr, tok_index))
        return tok_index

    tok("NULL") # unsed

    # start of module block
    tok("MODULE")      # MODULE "<name>"

    # can be used anywhere
    tok("REM")         #   REM "This is a comment"
    tok("TXT")         #   TXT "This is an annotation"
    tok("TAG")         #   TAG (i|f|n|o)<N> "This is an annotation"

    # header statements
    tok("DIMS")        #   DIMS i<max> f<max> n<max> o<max>
    tok("PI")          #   PI i<first> ... i<last>
    tok("PO")          #   PO o<first> ... o<last>

    # table block
    tok("TABLE")       #   TABLE ["<optional_table_name>"]
    tok("TAB_I")       #     TAB_I '<in_3state_pat>' TAB_F '<ff_4state_constr>' TAB_O '<out_4state_constr>'
    tok("TAB_F")
    tok("TAB_O")

    # circuit block
    tok("CIRCUIT")     #   CIRCUIT ["<optional_circuit_name>" ["<table_name>"... | "*"]]
    tok("OPS")         #     OPS NAND NOR
    tok("FFS")         #     FFS f<first> ... f<last>
    tok("NETS")        #     NETS n<first> ... n<last>
    tok("CNFN")        #     CNFN n1 i1 i2 NAND
    tok("FNCN")        #     FNCN n2 NOR n1 i3
    tok("CONN")        #     CONN n3 n1 n2
    tok("FUNC")        #     FUNC n3 NAND
    tok("NEXT")        #     NEXT f1 n3

    # end of module block
    tok("ENDMOD")      # ENDMOD

    # OP types. (LUT<N> is directly followed by 3-state LUT data, terminated by LUTEND)
    for s in """AND NAND OR NOR XOR XNOR ANDNOT ORNOT MUX NMUX
            AOI3 OAI3 AOI4 OAI4 LUT2 LUT3 LUT4 LUT5 LUT6 LUTEND""".split(): tok(s)

    for idx in range(1, cfg.num_pi+1): tok(f"i{idx}")
    for idx in range(1, cfg.num_ff+1): tok(f"f{idx}")
    for idx in range(1, cfg.num_op+1): tok(f"n{idx}")
    for idx in range(1, cfg.num_po+1): tok(f"o{idx}")

    vals = set()
    for w in product(*["01Z"  for _ in range(cfg.nbits_3state)]): vals.add("".join(w))
    for w in product(*["01ZX" for _ in range(cfg.nbits_4state)]): vals.add("".join(w))
    for l,w in sorted((len(v),v) for v in vals): tok(f"'{w.replace('X', '*').replace('Z', '-')}'")

    tok("STR")

    if cfg.shift_altgr:
        tok_shift = tok("SHIFT")
        tok_altgr = tok("ALTGR")

    def toks(ch, caps, alt):
        if caps == ' ': caps = "\\n"
        if alt == '\\': alt = "\\\\"
        if alt == '\"': alt = "\\\""
        if cfg.shift_altgr:
            tok(f'"{ch}" "{caps}" "{alt}"')
        else:
            tok(f'"{ch}"')
            tok(f'"{caps}"')
            tok(f'"{alt}"')

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
    toks(*"05_")
    toks(*"16`")
    toks(*"27{")
    toks(*"38|")
    toks(*"48}")
    toks(*"  ~")

    tok("STREND")

    return ret

if __name__ == "__main__":
    print()
    print("Large Example Token List")
    print("========================")
    cfg = GraphSprechConfig(
        nbits_3state = 4,
        nbits_4state = 4,
        shift_altgr = False
    )
    tokens = gentokens(cfg)
    tokens.pr_table(9)

    print()
    print("Medium Example Token List")
    print("=========================")
    cfg = GraphSprechConfig()
    tokens = gentokens(cfg)
    tokens.pr_table(10)

    print()
    print("Small Example Token List")
    print("========================")
    cfg = GraphSprechConfig(
        shift_altgr = True
    )
    tokens = gentokens(cfg)
    tokens.pr_table(8)
