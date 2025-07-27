import re, sys, types, itertools, collections
from dataclasses import dataclass, field

# import nltk
# nltk.download('words')
# en_basic_words = nltk.corpus.words.words("en-basic")
# len(en_basic_words) -> 850
en_basic_words = ["I", "a", "able", "about", "account", "acid", "across",
"act", "addition", "adjustment", "advertisement", "after", "again", "against",
"agreement", "air", "all", "almost", "among", "amount", "amusement", "and",
"angle", "angry", "animal", "answer", "ant", "any", "apparatus", "apple",
"approval", "arch", "argument", "arm", "army", "art", "as", "at", "attack",
"attempt", "attention", "attraction", "authority", "automatic", "awake",
"baby", "back", "bad", "bag", "balance", "ball", "band", "base", "basin",
"basket", "bath", "be", "beautiful", "because", "bed", "bee", "before",
"behaviour", "belief", "bell", "bent", "berry", "between", "bird", "birth",
"bit", "bite", "bitter", "black", "blade", "blood", "blow", "blue", "board",
"boat", "body", "boiling", "bone", "book", "boot", "bottle", "box", "boy",
"brain", "brake", "branch", "brass", "bread", "breath", "brick", "bridge",
"bright", "broken", "brother", "brown", "brush", "bucket", "building", "bulb",
"burn", "burst", "business", "but", "butter", "button", "by", "cake", "camera",
"canvas", "card", "care", "carriage", "cart", "cat", "cause", "certain",
"chain", "chalk", "chance", "change", "cheap", "cheese", "chemical", "chest",
"chief", "chin", "church", "circle", "clean", "clear", "clock", "cloth",
"cloud", "coal", "coat", "cold", "collar", "colour", "comb", "come", "comfort",
"committee", "common", "company", "comparison", "competition", "complete",
"complex", "condition", "connection", "conscious", "control", "cook", "copper",
"copy", "cord", "cork", "cotton", "cough", "country", "cover", "cow", "crack",
"credit", "crime", "cruel", "crush", "cry", "cup", "current", "curtain",
"curve", "cushion", "cut", "damage", "danger", "dark", "daughter", "day",
"dead", "dear", "death", "debt", "decision", "deep", "degree", "delicate",
"dependent", "design", "desire", "destruction", "detail", "development",
"different", "digestion", "direction", "dirty", "discovery", "discussion",
"disease", "disgust", "distance", "distribution", "division", "do", "dog",
"door", "doubt", "down", "drain", "drawer", "dress", "drink", "driving",
"drop", "dry", "dust", "ear", "early", "earth", "east", "edge", "education",
"effect", "egg", "elastic", "electric", "end", "engine", "enough", "equal",
"error", "even", "event", "ever", "every", "example", "exchange", "existence",
"expansion", "experience", "expert", "eye", "face", "fact", "fall", "false",
"family", "far", "farm", "fat", "father", "fear", "feather", "feeble",
"feeling", "female", "fertile", "fiction", "field", "fight", "finger", "fire",
"first", "fish", "fixed", "flag", "flame", "flat", "flight", "floor", "flower",
"fly", "fold", "food", "foolish", "foot", "for", "force", "fork", "form",
"forward", "fowl", "frame", "free", "frequent", "friend", "from", "front",
"fruit", "full", "future", "garden", "general", "get", "girl", "give", "glass",
"glove", "go", "goat", "gold", "good", "government", "grain", "grass", "great",
"green", "grey", "grip", "group", "growth", "guide", "gun", "hair", "hammer",
"hand", "hanging", "happy", "harbour", "hard", "harmony", "hat", "hate",
"have", "he", "head", "healthy", "hearing", "heart", "heat", "help", "here",
"high", "history", "hole", "hollow", "hook", "hope", "horn", "horse",
"hospital", "hour", "house", "how", "humour", "ice", "idea", "if", "ill",
"important", "impulse", "in", "increase", "industry", "ink", "insect",
"instrument", "insurance", "interest", "invention", "iron", "island", "jelly",
"jewel", "join", "journey", "judge", "jump", "keep", "kettle", "key", "kick",
"kind", "kiss", "knee", "knife", "knot", "knowledge", "land", "language",
"last", "late", "laugh", "law", "lead", "leaf", "learning", "leather", "left",
"leg", "let", "letter", "level", "library", "lift", "light", "like", "limit",
"line", "linen", "lip", "liquid", "list", "little", "living", "lock", "long",
"look", "loose", "loss", "loud", "love", "low", "machine", "make", "male",
"man", "manager", "map", "mark", "market", "married", "mass", "match",
"material", "may", "meal", "measure", "meat", "medical", "meeting", "memory",
"metal", "middle", "military", "milk", "mind", "mine", "minute", "mist",
"mixed", "money", "monkey", "month", "moon", "morning", "mother", "motion",
"mountain", "mouth", "move", "much", "muscle", "music", "nail", "name",
"narrow", "nation", "natural", "near", "necessary", "neck", "need", "needle",
"nerve", "net", "new", "news", "night", "no", "noise", "normal", "north",
"nose", "not", "note", "now", "number", "nut", "observation", "of", "off",
"offer", "office", "oil", "old", "on", "only", "open", "operation", "opinion",
"opposite", "or", "orange", "order", "organization", "ornament", "other",
"out", "oven", "over", "owner", "page", "pain", "paint", "paper", "parallel",
"parcel", "part", "past", "paste", "payment", "peace", "pen", "pencil",
"person", "physical", "picture", "pig", "pin", "pipe", "place", "plane",
"plant", "plate", "play", "please", "pleasure", "plough", "pocket", "point",
"poison", "polish", "political", "poor", "porter", "position", "possible",
"pot", "potato", "powder", "power", "present", "price", "print", "prison",
"private", "probable", "process", "produce", "profit", "property", "prose",
"protest", "public", "pull", "pump", "punishment", "purpose", "push", "put",
"quality", "question", "quick", "quiet", "quite", "rail", "rain", "range",
"rat", "rate", "ray", "reaction", "reading", "ready", "reason", "receipt",
"record", "red", "regret", "regular", "relation", "religion", "representative",
"request", "respect", "responsible", "rest", "reward", "rhythm", "rice",
"right", "ring", "river", "road", "rod", "roll", "roof", "room", "root",
"rough", "round", "rub", "rule", "run", "sad", "safe", "sail", "salt", "same",
"sand", "say", "scale", "school", "science", "scissors", "screw", "sea",
"seat", "second", "secret", "secretary", "see", "seed", "seem", "selection",
"self", "send", "sense", "separate", "serious", "servant", "sex", "shade",
"shake", "shame", "sharp", "sheep", "shelf", "ship", "shirt", "shock", "shoe",
"short", "shut", "side", "sign", "silk", "silver", "simple", "sister", "size",
"skin", "skirt", "sky", "sleep", "slip", "slope", "slow", "small", "smash",
"smell", "smile", "smoke", "smooth", "snake", "sneeze", "snow", "so", "soap",
"society", "sock", "soft", "solid", "some", "son", "song", "sort", "sound",
"soup", "south", "space", "spade", "special", "sponge", "spoon", "spring",
"square", "stage", "stamp", "star", "start", "statement", "station", "steam",
"steel", "stem", "step", "stick", "sticky", "stiff", "still", "stitch",
"stocking", "stomach", "stone", "stop", "store", "story", "straight",
"strange", "street", "stretch", "strong", "structure", "substance", "such",
"sudden", "sugar", "suggestion", "summer", "sun", "support", "surprise",
"sweet", "swim", "system", "table", "tail", "take", "talk", "tall", "taste",
"tax", "teaching", "tendency", "test", "than", "that", "the", "then", "theory",
"there", "thick", "thin", "thing", "this", "though", "thought", "thread",
"throat", "through", "thumb", "thunder", "ticket", "tight", "till", "time",
"tin", "tired", "to", "toe", "together", "tomorrow", "tongue", "tooth", "top",
"touch", "town", "trade", "train", "transport", "tray", "tree", "trick",
"trouble", "trousers", "true", "turn", "twist", "umbrella", "under", "unit",
"up", "use", "value", "verse", "very", "vessel", "view", "violent", "voice",
"waiting", "walk", "wall", "war", "warm", "wash", "waste", "watch", "water",
"wave", "wax", "way", "weather", "week", "weight", "well", "west", "wet",
"wheel", "when", "where", "while", "whip", "whistle", "white", "who", "why",
"wide", "will", "wind", "window", "wine", "wing", "winter", "wire", "wise",
"with", "woman", "wood", "wool", "word", "work", "worm", "wound", "writing",
"wrong", "year", "yellow", "yes", "yesterday", "you", "young"]

example_text = """
REM "Tokenizer encode/decode example text."

PROMPT "What is 7+4?"
TXT "11."

PROMPT "What's a short name for Robert?"
TXT "Bob."

MODULE "test_1"
  DIMS i16 f8 n64 o4
  PI i1 i2 i3 i4
  PO o1 o2

  TABLE "ref"
    '-------- 11-------------- XXXXXXXX 1-XX'
    '-------- --00------------ XXXXXXXX -0XX'
    '-------- ---------------- XXXXXXXX 01XX'

  CIRCUIT "impl" "ref"
    FNCN o1 AND i1 i2
    FNCN o2 OR i3 i4

PROMPT ```
Create another truth table, that explicitly encodes the
cases where the AND-inputs are '1' and the OR-inputs are '0'.
```
REM ```
Thinking.. The user asks me to ... the TABLE "ref" has one line
per gate, plus a final line with defaults ... in order to get
the encoding the user asks for I therefore should ...
```

  TABLE "alt"
    '-------- 0--------------- XXXXXXXX 0-XX'
    '-------- -0-------------- XXXXXXXX 0-XX'
    '-------- --1------------- XXXXXXXX -1XX'
    '-------- ---1------------ XXXXXXXX -1XX'
    '-------- ---------------- XXXXXXXX 10XX'

TXT "Finished creating the table."

ENDMOD
"""

re_keywords = re.compile("""
(?<![a-zA-Z]) ( PROMPT | MODULE | REM | TXT | TAG | DIMS | PI | PO | TABLE |
CIRCUIT | OPS | FFS | NETS | CNFN | FNCN | CONN | FUNC | NEXT | ENDMOD |
AND | NAND | OR | NOR | XOR | XNOR | ANDNOT | ORNOT | MUX | NMUX |
AOI3 | OAI3 | AOI4 | OAI4 | LUT2 | LUT3 | LUT4 | LUT5 | LUT6 | [ifno][0-9]+ ) (?![a-zA-Z])
""", re.A|re.X)

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


@dataclass
class GraphSprechConfig:
    num_pi: int = 16
    num_ff: int = 8
    num_op: int = 64
    num_po: int = 4
    nbits_3state: tuple = (2,)
    nbits_4state: tuple = (2,)
    shift_altgr: bool = False
    with_words: bool = False
    with_dbls: bool = False
    with_tris: bool = False

@dataclass
class TokenList:
    lines: list = field(default_factory=list)
    tokens: dict = field(default_factory=dict)
    encoder: dict = field(default_factory=dict)
    decoder: list = field(default_factory=list)
    pi_offset: int = 0
    ff_offset: int = 0
    op_offset: int = 0
    po_offset: int = 0

    def pr_table(self, cols=5, /):
        col_height = (len(self.lines)+cols-1) // cols
        col_widths = [0]*cols

        for i in range(col_height):
            for j in range(cols):
                k = i + col_height * j
                l = self.lines[k] if k < len(self.lines) else ""
                col_widths[j] = max(col_widths[j], len(l))

        for i in range(col_height):
            for j in range(cols):
                k = i + col_height * j
                l = self.lines[k] if k < len(self.lines) else ""
                print(f"{l}\n" if j == cols-1 else f"{l:<{col_widths[j]}} | ", end="")

def quote_str_char(c):
    if c == '"': return '"\\""'
    if c == '\n': return '"\\n"'
    if c == '\\': return '"\\\\"'
    return f'"{c}"'

def gentokens(cfg: GraphSprechConfig = GraphSprechConfig()):
    ret = TokenList()
    tok_shift = None
    tok_altgr = None

    def tok(x):
        tok_index = len(ret.lines)
        ret.lines.append(f"{tok_index:<3} {x}")
        ret.decoder.append(x)
        ret.tokens[x] = tok_index
        def enc(x, y):
            ret.encoder[x] = y
        if x.startswith('"'):
            s = x.removeprefix('"')
            s = s.removesuffix('"')
            s = s.replace('" "', '')
            s = s.replace('\\"', '"')
            s = s.replace('\\n', '\n')
            s = s.replace('\\\\', '\\')
            s = tuple(quote_str_char(c) for c in s)
        else:
            s = (x,)
        if len(s) >= 1: enc(s[0], (tok_index,))
        if len(s) >= 2: enc(s[1], (tok_shift, tok_index))
        if len(s) >= 3: enc(s[2], (tok_altgr, tok_index))
        return tok_index

    # only as EOF and ERROR marker
    tok("NULL")

    # a break is just an empty line
    tok("BREAK")

    # chatbot interface:  PROMPT "<text>"
    # (response as TXT in the next line and/or REM later)
    tok("PROMPT")

    # start of module block
    tok("MODULE")      # MODULE "<name>"

    # can be used anywhere
    tok("REM")         #   REM "This is a comment or remark that can be ignored"
    tok("TXT")         #   TXT "This is a relevant information or reply to a prompt"
    tok("TAG")         #   TAG (i|f|n|o)<N> "This is an annotation of that entity"

    # header statements
    tok("DIMS")        #   DIMS i<max> f<max> n<max> o<max>
    tok("PI")          #   PI i<first> ... i<last>
    tok("PO")          #   PO o<first> ... o<last>

    # table block
    tok("TABLE")       #   TABLE ["<optional_table_name>" ["<table_or_circuit_name>"... | "*"]]
    tok("LUT_Q")       #     '<ff_3state_pat> <in_3state_pat> <ff_4state_constr> <out_4state_constr>'
    tok("LUT_I")       #     ^LUT_Q          ^LUT_I          ^LUT_D             ^LUT_O              ^LUT_E
    tok("LUT_D")
    tok("LUT_O")
    tok("LUT_E")

    # circuit block
    tok("CIRCUIT")     #   CIRCUIT ["<optional_circuit_name>" ["<table_or_circuit_name>"... | "*"]]
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

    # OP types. (LUT<N> is directly followed by 3-state LUT data, terminated by LUT_E)
    for s in """AND NAND OR NOR XOR XNOR ANDNOT ORNOT MUX NMUX
            AOI3 OAI3 AOI4 OAI4 LUT2 LUT3 LUT4 LUT5 LUT6""".split(): tok(s)

    tok("STR_B")  # start of "..." string
    tok("STR_Q")  # start of ```\n...\n```\n string
    tok("STR_E")  # end of string

    for idx in range(1, cfg.num_pi+1): tok(f"i{idx}")
    for idx in range(1, cfg.num_ff+1): tok(f"f{idx}")
    for idx in range(1, cfg.num_op+1): tok(f"n{idx}")
    for idx in range(1, cfg.num_po+1): tok(f"o{idx}")

    vals = set("01ZX")
    for n in cfg.nbits_3state:
        for w in itertools.product(*["01Z" for _ in range(n)]): vals.add("".join(w))
    for n in cfg.nbits_4state:
        for w in itertools.product(*["01ZX" for _ in range(n)]): vals.add("".join(w))
    for l,w in sorted((len(v),v) for v in vals):
        tok(f"'{w.replace('Z', '-')}'")

    if cfg.shift_altgr:
        tok_shift = tok("SHIFT")
        tok_altgr = tok("ALTGR")

    if cfg.with_words:
        if not tok_shift:
            tok_shift = tok("SHIFT")
        tok("CAPS")

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
    toks(*"05`")
    toks(*"16{")
    toks(*"27|")
    toks(*"38}")
    toks(*"49~")
    toks(*"  _")

    lex = []
    if cfg.with_words:
        lex += [f"_{w}" for w in en_basic_words]

    if cfg.with_dbls:
        dbls = set()
        for w in en_basic_words:
            if len(w) <= 2: continue
            for i in range(0,len(w)-1):
                dbls.add(w[i:i+2])
        for dbl in dbls:
            lex.append(f".{dbl}")

    if cfg.with_tris:
        tris = set()
        for w in en_basic_words:
            if len(w) <= 3: continue
            for i in range(0,len(w)-2):
                tris.add(w[i:i+3])
        for tri in tris:
            lex.append(f".{tri}")

    if lex:
        for _,_,t in sorted((len(t), t[1:]+t[0], t) for t in lex):
            tok(t)

    ret.pi_offset = ret.tokens["i1"]
    ret.ff_offset = ret.tokens["f1"]
    ret.op_offset = ret.tokens["n1"]
    ret.po_offset = ret.tokens["o1"]

    return ret

def encode(lex, text):
    tokens = []
    state = ""
    pos = 0

    while pos < len(text):
        if not state:
            if text[pos:pos+2] == "\n\n":
                pos += 1
                tokens += lex.encoder["BREAK"]
                continue
            if text[pos] in " \t\n":
                pos += 1
                continue
            if m := re_keywords.match(text, pos):
                pos += len(m[0])
                assert m[0] in lex.encoder
                tokens += lex.encoder[m[0]]
                continue
            if text[pos] == '"':
                pos += 1
                state = 'STR_B'
                tokens += lex.encoder['STR_B']
                continue

        elif state == 'STR_B':
            if text[pos] == '"':
                state = ""; t = 'STR_E'
            elif text[pos:pos+2] == '\\n':
                pos += 1; t ='"\\n"'
            elif text[pos:pos+2] in ('\\"', '\\\\'):
                pos += 1; t ='"\\{text[pos]}"'
            else:
                t = quote_str_char(text[pos])
            assert t in lex.encoder, f"Token {t} not in encoder table."
            tokens += lex.encoder[t]
            pos += 1
            continue

        tokens.append(0)
        break

    return tokens

def decode(lex, toks):
    return "FIXME"

def tok2str(lex, toks):
    if isinstance(toks, int):
        s = lex.decoder[toks]
        if ' ' in s and s != '" "':
            s = s.removeprefix('\"')
            s = s.removesuffix('\"')
            s = s.replace('" "', '')
            s = s.replace('\\"', '"')
            s = f"[{s}]"
        return s
    if toks is None:
        return "None"
    return " ".join(tok2str(lex, t) for t in toks)

def main():
    if args and args[0] == "-t":
        cfg = GraphSprechConfig(shift_altgr = True)
        lex = gentokens(cfg)
        lex.pr_table(8)
        for s in args[1:]:
            print()
            print(f"Input: {s}")
            t = encode(lex, s)
            print(f"Ids: {' '.join(str(i) for i in t)}")
            print(f"Tokens: {tok2str(lex, t)}")
            x = decode(lex, t)
            print(f"Output: {x}")
        return

    print()
    print("Huge Example Token List")
    print("=======================")
    cfg = GraphSprechConfig(
        nbits_3state = (4,),
        nbits_4state = (2,3,4),
        shift_altgr = False,
        with_words = True,
        with_dbls = True,
        with_tris = True
    )
    lex = gentokens(cfg)
    lex.pr_table(8)

    print()
    print("Large Example Token List")
    print("========================")
    cfg = GraphSprechConfig(
        nbits_3state = (4,),
        nbits_4state = (4,),
        shift_altgr = False
    )
    lex = gentokens(cfg)
    lex.pr_table(9)

    print()
    print("Medium (Default) Token List")
    print("===========================")
    cfg = GraphSprechConfig()
    lex = gentokens(cfg)
    lex.pr_table(10)

    print()
    print("Small Example Token List")
    print("========================")
    cfg = GraphSprechConfig(
        shift_altgr = True
    )
    lex = gentokens(cfg)
    lex.pr_table(8)

if __name__ == "__main__":
    cmdname, *args = sys.argv
    sys.exit(main())
