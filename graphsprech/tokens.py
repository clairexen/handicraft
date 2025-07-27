import types
from itertools import product

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
    with_words: bool = False

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

    vals = set("01ZX")
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

    if cfg.with_words:
        for w in en_basic_words:
            tok(f"_{w}")

    tok("STREND")

    return ret

if __name__ == "__main__":
    print()
    print("Huge Example Token List")
    print("=======================")
    cfg = GraphSprechConfig(
        nbits_3state = 4,
        nbits_4state = 4,
        shift_altgr = False,
        with_words = True
    )
    tokens = gentokens(cfg)
    tokens.pr_table(6)

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
