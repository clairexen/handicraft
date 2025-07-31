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
class LogiChatConfig:
    gates: tuple = ("BUF", "NOT", "AND", "NAND", "OR", "NOR", "XOR", "XNOR", "ANDNOT", "ORNOT",
            "MUX", "NMUX", "AOI3", "OAI3", "AOI4", "OAI4", "LUT2", "LUT3", "LUT4", "LUT5", "LUT6")
    idx_base: int = 8
    max_nbits: int = 2
    shift_altgr: bool = False
    with_words: bool = False
    with_dbls: bool = False
    with_tris: bool = False

@dataclass
class TokenList:
    cfg: LogiChatConfig = field(default_factory=LogiChatConfig)
    lines: list = field(default_factory=list)
    tokens: dict = field(default_factory=dict)
    encoder: dict = field(default_factory=dict)
    decoder: dict = field(default_factory=dict)
    pi_offset: int = 0
    ff_offset: int = 0
    op_offset: int = 0
    po_offset: int = 0

    def pr_table(self, cols=5, /):
        lines = [l for l in self.lines if l is not None]

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
                print(f"{l}\n" if j == cols-1 else f"{l:<{col_widths[j]}} | ", end="")

        print(f"({len(lines)} tokens in total)")

    def finish(self):
        self.pi_offset = self.tokens["i1"]
        self.ff_offset = self.tokens["f1"]
        self.op_offset = self.tokens["n1"]
        self.po_offset = self.tokens["o1"]

        opnames = " | ".join(t for t in self.cfg.gates)
        objnames = " | ".join(t for t in self.decoder.values() if t[0] in 'iqnodfa')
        self.re_keywords = re.compile(f"""
            (?<![a-zA-Z0-9]) ( PROMPT | REPLY | QUERY | REM | TXT | TAG | MODULE | DIMS | PI | PO |
                    TABLE | PTABLE | GET | SET | CIRCUIT | OPS | DEF | ENDMOD | {opnames} | {objnames}) (?![a-zA-Z0-9])
        """, re.A|re.X)

def quote_str_char(c):
    if c == '"': return '"\\""'
    if c == '\n': return '"\\n"'
    if c == '\\': return '"\\\\"'
    if ord(c) < 31 or ord(c) >= 127:
        return f'"\\x{ord(c):02x}"'
    return f'"{c}"'

def gentokens(cfg: LogiChatConfig = LogiChatConfig()):
    ret = TokenList()
    ret.cfg = cfg
    tok_shift = None
    tok_altgr = None

    def tok(x, tok_index=None):
        if tok_index is None:
            tok_index = len(ret.lines)
            ret.lines.append(None)
        else:
            if isinstance(tok_index, str):
                tok_index = ascii_ctrls_by_name[tok_index]
            assert ret.lines[tok_index] is None
        ret.lines[tok_index] = f"{tok_index:<3} {x}"
        ret.decoder[tok_index] = x
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

    # skip 7-bit ASCII range + 2 reserved slots for now
    while len(ret.lines) < (128 + 2):
        ret.lines.append(None)

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

    # can be used anywhere
    tok("#", "DLE")    # #-comments at the end of a line
    tok("REM", "STX")  #   REM "This is just a remark that can be ignored"
    tok("TXT", "ETX")  #   TXT "This is relevant information and/or reply to a prompt or query"
    tok("TAG", "SUB")  #   TAG (i|f|n|o)<N> "This is an annotation of that entity"

    # start of module block
    tok("MODULE", 128) # MODULE ["<optional_name>"]

    # header statements
    tok("DIMS")        #   DIMS i<max> d<max> n<max> o<max> q<max> f<max> a<max>
    tok("PI")          #   PI i<first> ... i<last>
    tok("PO")          #   PO o<first> ... o<last>

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
    for s in cfg.gates: tok(s)

    tok("STR_B", "DC3")  # normal "..."-strings: STR_B ... STR_E
    tok("STR_E", "DC4")  # here-doc-style strings: STR_B STR_B ... STR_E STR_E

    for idx in range(cfg.idx_base): tok(f"i{idx}")
    for idx in range(cfg.idx_base): tok(f"q{idx}")
    for idx in range(cfg.idx_base): tok(f"n{idx}")
    for idx in range(cfg.idx_base): tok(f"o{idx}")
    for idx in range(cfg.idx_base): tok(f"d{idx}")
    for idx in range(cfg.idx_base): tok(f"f{idx}")
    for idx in range(cfg.idx_base): tok(f"a{idx}")
    for idx in range(cfg.idx_base): tok(f"x{idx}")

    vals = set("01ZX")
    for n in range(1, cfg.max_nbits+1):
        for w in itertools.product(*["01ZX" for _ in range(n)]):
            vals.add("".join(w))
    for l,w in sorted((len(v),v) for v in vals):
        tok(f"'{w.replace('Z', '-')}'")

    if cfg.shift_altgr or cfg.with_words:
        tok_shift = tok("SHIFT", "SO")
    if cfg.shift_altgr:
        tok_altgr = tok("ALTGR", "SI")
    if cfg.with_words:
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
        if cfg.shift_altgr:
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

    ret.finish()
    return ret

def encode(lex, text, encodeText=True):
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
                    t += lex.encoder["STR_E"]
                elif state_str2:
                    while text[pos+l:pos+l+4] != "\n``\n" and pos+l < len(text):
                        l += 1
                    t += lex.encoder["STR_E"]
                    t += lex.encoder["STR_E"]
                else:
                    assert state_str3
                    while text[pos+l:pos+l+5] != "\n'''\n" and pos+l < len(text):
                        l += 1
                    t += lex.encoder["STR_E"]
                    t += lex.encoder["STR_E"]
                    t += lex.encoder["STR_E"]
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
                    t = quote_str_char(text[pos])
                assert t in lex.encoder, f"Token {t} not in encoder table."
                tokens += lex.encoder[t]
                pos += 1
                continue

            elif state_str2:
                if text[pos:].startswith('\n``\n'):
                    state_str2 = False; t = 'STR_E'
                    tokens += lex.encoder[t]
                    pos += 3
                else:
                    t = quote_str_char(text[pos])
                assert t in lex.encoder, f"Token {t} not in encoder table."
                tokens += lex.encoder[t]
                pos += 1
                continue

            else:
                assert state_str3
                if text[pos:].startswith("\n'''\n"):
                    state_str2 = False; t = 'STR_E'
                    tokens += lex.encoder[t]
                    pos += 3
                else:
                    t = quote_str_char(text[pos])
                assert t in lex.encoder, f"Token {t} not in encoder table."
                tokens += lex.encoder[t]
                pos += 1
                continue

        if state_bits:
            if text[pos] == "'":
                state_bits = False; pos += 1
                tokens += lex.encoder["LUT_E"]
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
            tokens += lex.encoder[t]
            continue

        # no special state
        if text[pos:pos+2] == "\n\n":
            pos += 1
            tokens += lex.encoder["BREAK"]
            continue
        if text[pos] in " \t\n":
            pos += 1
            continue
        if m := lex.re_keywords.match(text, pos):
            pos += len(m[0])
            assert m[0] in lex.encoder
            tokens += lex.encoder[m[0]]
            continue
        if text[pos] == '"':
            pos += 1
            state_str1 = True
            tokens += lex.encoder['STR_B']
            continue
        if text[pos:].startswith("``\n"):
            pos += 3
            state_str2 = True
            tokens += lex.encoder['STR_B']
            tokens += lex.encoder['STR_B']
            continue
        if text[pos:].startswith("'''\n"):
            pos += 4
            state_str3 = True
            tokens += lex.encoder['STR_B']
            tokens += lex.encoder['STR_B']
            tokens += lex.encoder['STR_B']
            continue
        if text[pos] == "'":
            pos += 1
            state_bits = True
            tokens += lex.encoder['LUT_B']
            continue

        if text[pos] == "(":
            pos += 1
            tokens += lex.encoder['FUN_B']
            continue

        if text[pos] == ")":
            pos += 1
            tokens += lex.encoder['FUN_E']
            continue

        tokens.append(0)
        tokens.append(text[pos:pos+15])
        break

    return tokens

def decode(lex, tokens):
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
        tok = lex.decoder[tokens[pos]]
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

        if tok[0] in "iqnodfa" or tok in lex.cfg.gates:
            if last_tok != "FUN_B":
                text.append(f' {tok}')
            else:
                text.append(tok)
            continue

        if tok == 'STR_B':
            state_str = True
            text.append(' ')
            if tokens[pos-1] == tokens[pos]:
                if tokens[pos-1] == tokens[pos+1]:
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
                if tokens[pos-1] == tokens[pos+1]:
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

        pos -= 1
        text.append(f"*** DECODE ERROR AT POSITION {pos}: {tok} ***")
        break

    return "".join(text)

def tok2str(lex, toks):
    if isinstance(toks, str):
        return repr(toks)
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
        if len(args) == 1:
            args.append(example_text)
        cfg = LogiChatConfig(
            shift_altgr = False
        )
        lex = gentokens(cfg)
        lex.pr_table(8)
        for s in args[1:]:
            print()
            print(f"Input: {s}")
            t = encode(lex, s, False)
            print(f"Ids: {' '.join(repr(i) for i in t)}")
            print(f"Tokens: {tok2str(lex, t)}")
            x = decode(lex, t)
            print(f"Output: {x}")
        return

    print()
    print("Large Example Token List")
    print("========================")
    cfg = LogiChatConfig(
        max_nbits = 4,
        with_words = True,
        with_dbls = True,
        with_tris = True
    )
    lex = gentokens(cfg)
    lex.pr_table(8)

    print()
    print("Medium Example Token List")
    print("=========================")
    cfg = LogiChatConfig(
        max_nbits = 4,
    )
    lex = gentokens(cfg)
    lex.pr_table(9)

    print()
    print("Small (Default) Token List")
    print("==========================")
    cfg = LogiChatConfig()
    lex = gentokens(cfg)
    lex.pr_table(10)

    print()
    print("Tiny Example Token List")
    print("=======================")
    cfg = LogiChatConfig(
        shift_altgr = True
    )
    lex = gentokens(cfg)
    lex.pr_table(8)

if __name__ == "__main__":
    cmdname, *args = sys.argv
    sys.exit(main())
