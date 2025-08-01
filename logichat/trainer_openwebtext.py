import datasets, pickle, numpy, sys, os
import config, tokens, utils
from pathlib import Path
from subprocess import Popen, PIPE

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

datafile_parts = []
datafile_partidx = 0
datafile_partpipes = []
datafile_bytes = None

def partpipe_close(N=4):
    global datafile_parts, datafile_partidx, datafile_partpipes, datafile_bytes
    if datafile_bytes is not None:
        datafile_partpipes[-1].stdin.close()
        print(".")
    while len(datafile_partpipes) > N:
        datafile_partpipes[0].wait()
        del datafile_partpipes[0]
    datafile_bytes = None

def partpipe_write(t):
    global datafile_parts, datafile_partidx, datafile_partpipes, datafile_bytes
    if datafile_bytes is None or datafile_bytes > 1024*1024:
        partpipe_close()
        datafile_parts.append(str(datafile.with_suffix(f".part{datafile_partidx:05d}")))
        print(f" `- writing {datafile_parts[-1]}", flush=True, end="")
        datafile_partpipes.append(Popen(["/bin/sh", "-c", f"python3 tokens.py -e '{datafile_parts[-1]}'"], stdin=PIPE))
        datafile_partidx += 1
        datafile_bytes = 0
    datafile_partpipes[-1].stdin.write(bytes(t, "ascii"))
    datafile_bytes += len(t)

total = 0
rejected = 0
print(f"\nWriting {datafile} ...")
while meta["index"] < len(dataset) and total - rejected < limit:
    t = dataset[meta["index"]]["text"]
    total += 1; meta["index"] += 1
    if all(31 < ord(c) < 127 for c in t if c != "\n"):
        t = f"REM '''\n{t}\n'''\n\x00"
        partpipe_write(t)
    else:
        rejected += 1

print(f" `- copy parts to single output file and remove parts.")
with datafile.open("wb") as f:
    for fn in datafile_parts:
        f.write(open(fn, "rb").read())
        os.remove(fn)

print(f"  written {total - rejected} / {total} items (= {100*(total-rejected) // total}%)")

partpipe_close(0)
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
