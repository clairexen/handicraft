import datasets, pickle, numpy, sys, os
import config, tokens
cfg = config.cfg_small
lex = cfg.lex()

from pathlib import Path
Path.mkdirs = lambda self: self.mkdir(parents=True, exist_ok=True)

dataset = datasets.load_dataset("openwebtext")
dataset = dataset["train"].train_test_split(test_size=0.001, seed=1357, shuffle=True)

datapath = Path(f"datasets/openwebtext_{cfg.name}")
datapath.mkdir(parents=True, exist_ok=True)

with datapath.joinpath('meta.pkl').open('wb') as f:
    pickle.dump(lex.meta, f)

total = 0
rejected = 0
for fname, name in (("val", "test"), ("train", "train")):
    datafile = datapath.joinpath(f"{fname}.{lex.binext}")
    print(f"Writing {datafile} ...")
    with datafile.open("wb") as f:
        cnt = 0
        for d in dataset[name]:
            if cnt == 50000: break
            total += 1
            t = d["text"]
            if all(31 < ord(c) < 127 for c in t if c != "\n"):
                t = f"REM '''\n{t}\n'''\n"
                t = lex.encode(t) + [0]
                t = numpy.array(t, lex.bintype)
                t.tofile(f)
                cnt += 1
            else:
                rejected += 1
        print(f"  written {cnt} items")

print(f"Rejected {rejected} / {total} items containing non-ASCII chars.")
