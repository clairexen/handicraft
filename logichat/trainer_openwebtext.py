import datasets, pickle, numpy, sys, os
import config, tokens, utils

dataset_name = "openwebtext"

opts, args = utils.opts_args()

if len(args) == 0:
    args.append(config.cfg.name)

if len(args) == 1:
    for i in range(6):
        os.system(f"set -x; python3 '{sys.argv[0]}' {args[0]} {i}")
    sys.exit()

if len(args) == 2:
    args.append("")

cfg_name, partnum, mblimit = args
partnum = int(partnum)
if not mblimit or not int(mblimit):
    mblimit = 100 if partnum else 10
limit = 350*int(mblimit)

cfg = config.cfgs[cfg_name][0]
lex = cfg.lex()

from pathlib import Path
Path.mkdirs = lambda self: self.mkdir(parents=True, exist_ok=True)

datasets = datasets.load_dataset("openwebtext")
datasets = datasets["train"].train_test_split(test_size=50000, seed=1357, shuffle=True)

datapath = Path(f"datasets/{dataset_name}")
datapath.mkdir(parents=True, exist_ok=True)

metafile = datapath.joinpath('meta.pkl')
if partnum:
    with metafile.open('rb') as f:
        meta = pickle.load(f)
else:
    meta = lex.meta
    meta["index"] = 0
    with metafile.open('wb') as f:
        pickle.dump(meta, f)

print(f"\nInitial Meta:")
for key in sorted(meta.keys()):
    if key in ("itos", "stoi"):
        print(f"    {key:<10} {repr(meta[key])[:60]} ....")
    else:
        print(f"    {key:<10} {repr(meta[key])}")

if partnum:
    dataset = datasets["train"]
    datafile = datapath.joinpath(f"{cfg_name}.t{partnum:02d}.{lex.binext}")
else:
    dataset = datasets["test"]
    datafile = datapath.joinpath(f"{cfg_name}.val.{lex.binext}")

total = 0
rejected = 0
print(f"\nWriting {datafile} ...")
with datafile.open("wb") as f:
    while meta["index"] < len(dataset) and total - rejected < limit:
        t = dataset[meta["index"]]["text"]
        total += 1; meta["index"] += 1
        if all(31 < ord(c) < 127 for c in t if c != "\n"):
            t = f"REM '''\n{t}\n'''\n"
            t = lex.encode(t) + [0]
            t = numpy.array(t, lex.bintype)
            t.tofile(f)
        else:
            rejected += 1
    print(f"  written {total - rejected} / {total} items (= {100*(total-rejected) // total}%)")

print(f"Rejected {rejected} / {total} items containing non-ASCII chars.")

print(f"\nFinal Meta:")
for key in sorted(meta.keys()):
    if key in ("itos", "stoi"):
        print(f"    {key:<10} {repr(meta[key])[:60]} ....")
    else:
        print(f"    {key:<10} {repr(meta[key])}")

with metafile.open('wb') as f:
    pickle.dump(meta, f)

print(f"\n{'='*70}\n")
