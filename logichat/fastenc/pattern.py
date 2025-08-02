import sys, os

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, parent_dir)

import config
cfg = config.cfg
lex = cfg.lex()

quote_and_sort_by_len = lambda l: [f'"{t}"' for _,_,t in sorted((-len(t), t.lower(), t) for t in set(l))]

re_words = "|".join(quote_and_sort_by_len([
    *[t[1:] for t in lex.decoder.values() if t.startswith("_")],
    *[t[1].upper() + t[2:] for t in lex.decoder.values() if t.startswith("_") and t != "_I"],
    *[t[1:].upper() for t in lex.decoder.values() if t.startswith("_") and t != "_I"]
]))

re_frags = "|".join(quote_and_sort_by_len([
    *[t[1:] for t in lex.decoder.values() if t.startswith(".")],
    *[t[1].upper() + t[2:] for t in lex.decoder.values() if t.startswith(".")],
    *[t[1:].upper() for t in lex.decoder.values() if t.startswith(".")]
]))

with open("pattern.re", "w") as f:
    f.write(f"""
typedef char YYCTYPE;

int match_word(const char *s) {{
    const char *YYCURSOR = s, *YYMARKER;
    /*!re2c
        re2c:yyfill:enable = 0;

        {re_words}  {{ return YYCURSOR - s; }}
        *  {{ return -1; }}
    */
}}
int match_frag(const char *s) {{
    const char *YYCURSOR = s;
    /*!re2c
        re2c:yyfill:enable = 0;

        {re_frags}  {{ return YYCURSOR - s; }}
        *  {{ return -1; }}
    */
}}
""")
