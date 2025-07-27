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

dims = (16, 8, 64, 4)
shift_alt = False
tok_index = 0

def tok(x):
    global tok_index
    tok_index += 1
    print(f"{tok_index:<3} {x}")

tok("FUNCTION")
tok("LUTI")
tok("LUTF")
tok("LUTO")
tok("REM")
tok("TXT")
tok("CIRCUIT")
tok("CNFN")
tok("FNCN")
tok("CONN")
tok("FUNC")
tok("END")

for s in """AND NAND OR NOR XOR XNOR ANDNOT ORNOT MUX NMUX
            AOI3 OAI3 AOI4 OAI4 LUT2 LUT3 LUT4 LUT5 LUT6""".split():
    tok(s)

def toks(ch, caps, alt):
    if caps == ' ': caps = "\\n"
    if alt == '\\': alt = "\\\\"
    if alt == '\"': alt = "\\\""
    if shift_alt:
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

if shift_alt:
    tok("SHIFT")
    tok("ALT")

for idx in range(1, dims[0]+1): tok(f"i{idx}")
for idx in range(1, dims[1]+1): tok(f"f{idx}")
for idx in range(1, dims[2]+1): tok(f"n{idx}")
for idx in range(1, dims[3]+1): tok(f"o{idx}")

for w in ("".join(w) for w in product("01XZ", "01XZ", "01XZ", "01XZ")):
    tok(f"h{w} '{w.replace('X', '*').replace('Z', '-')}'")
