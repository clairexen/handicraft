import sys, os

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, parent_dir)

import config
cfg = config.cfg
lex = cfg.lex()

with open("pattern.re", "w") as f:
    f.write("/*!rules:re2c:pat_keywords\n")
    for i,s in enumerate(lex.token_names):
        if not s or s[0] in "._'\"#": continue
        f.write(f"  \"{s}\" {{ pushtok({i}); continue; }}\n")
    f.write("*/\n")

    f.write("/*!rules:re2c:pat_quoted_keywords\n")
    for i,s in enumerate(lex.token_names):
        if not s or s[0] in "._'\"#": continue
        f.write(f"  \"'{s}'\" {{ pushtok({i}); continue; }}\n")
    f.write("*/\n")

    f.write("/*!rules:re2c:pat_strtoks\n")
    for s,i in lex.stoi_map.items():
        if len(s) <= 1: continue
        f.write(f"  \"{s}\" {{ pushtok({i}); continue; }}\n")
    f.write("*/\n")

with open("pattern.cc", "w") as f:
    for i,s in enumerate(lex.token_names):
        if not s or s[0] in "._'\"#": continue
        f.write(f"#define TOK_{s} {i}\n")

    f.write("const char *toknames[] = {")
    for n in lex.token_names:
        if n is None:
            f.write(f"  NULL,\n")
            continue
        n = n.replace('\\', '\\\\')
        n = n.replace('"', '\\"')
        f.write(f"  \"{n}\",\n")
    f.write("};")
