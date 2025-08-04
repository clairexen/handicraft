import sys, os

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, parent_dir)

import config
cfg = config.cfg
lex = cfg.lex()

with open("pattern.re", "w") as f:
    f.write("/*!re2c\n")
    bag = [f'"{s}"' for s in lex.stoi_map.keys() if len(s) > 1]
    f.write(f"  strtok = {'|\n'.join(bag)};")
    f.write("*/\n")

    f.write("/*!rules:re2c:pat_keywords\n")
    for s,i in lex.stoi_map.items():
        if len(s) <= 1: continue
        f.write(f"  \"{s}\" {{ pushtok({i}); continue; }}\n")
    f.write("*/\n")
