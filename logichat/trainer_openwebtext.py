import datasets, pickle, numpy, sys, os
import config, tokens, utils
from pathlib import Path
from subprocess import Popen, PIPE

dataset_name = "openwebtext"

opts, args = utils.opts_args()

if len(args) == 0:
    args.append(config.cfg.name)

if len(args) == 1:
    os.system(f"set -x; python3 '{sys.argv[0]}' {args[0]} test")
    os.system(f"set -x; python3 '{sys.argv[0]}' {args[0]} train")
    sys.exit()

if len(args) == 2:
    args.append("")

cfg_name, split, mblimit = args
if not mblimit or not int(mblimit):
    mblimit = 100 if split == "train" else 10
limit = 350*int(mblimit)

cfg = config.cfgs[cfg_name][0]
lex = cfg.lex()

Path.mkdirs = lambda self: self.mkdir(parents=True, exist_ok=True)

datasets = datasets.load_dataset("openwebtext")
datasets = datasets["train"].train_test_split(test_size=50000, seed=1357, shuffle=True)

datapath = Path(f"datasets/{dataset_name}")
datapath.mkdir(parents=True, exist_ok=True)

metafile = datapath.joinpath(cfg_name + '.meta.pkl')
if split == "train":
    with metafile.open('rb') as f:
        meta = pickle.load(f)
else:
    meta = lex.meta
    with metafile.open('wb') as f:
        pickle.dump(meta, f)

print(f"\nInitial Meta:")
for key in sorted(meta.keys()):
    if key in ("itos", "stoi"):
        print(f"    {key:<10} {repr(meta[key])[:60]} ....")
    else:
        print(f"    {key:<10} {repr(meta[key])}")

if split == "train":
    dataset = datasets["train"]
    datafile = datapath.joinpath(f"{cfg_name}.train.{lex.binext}")
else:
    dataset = datasets["test"]
    datafile = datapath.joinpath(f"{cfg_name}.test.{lex.binext}")

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

def convert_special_chars(t):
    t = t.replace("\u00a9", '(C)')
    t = t.replace("\u00ad", '')
    t = t.replace("\u00ae", '(R)')
    t = t.replace("\u00b4", "'")
    t = t.replace("\u200b", '')
    t = t.replace("\u2011", '-')
    t = t.replace("\u2013", '-')
    t = t.replace("\u2014", '-')
    t = t.replace("\u2015", '-')
    t = t.replace("\u2019", "'")
    t = t.replace("\u201c", '"')
    t = t.replace("\u201d", '"')
    t = t.replace("\u201f", '"')
    t = t.replace("\u2026", '...')
    t = t.replace("\uff01", '!')
    t = t.replace("\uff08", '(')
    t = t.replace("\uff09", ')')
    t = t.replace("\uff0d", '"')
    return t

total = 0
rejected = 0
special_chars_cnt = dict()
print(f"\nWriting {datafile} ...")
while total < len(dataset) and total - rejected < limit:
    t = dataset[total]["text"]; total += 1
    if all(31 < ord(c) < 127 for c in t if c != "\n"):
        t = f"REM '''\n{convert_special_chars(t)}\n'''\n\x00"
        partpipe_write(t)
    else:
        special_chars = [c for c in t if (ord(c) < 32 or 127 <= ord(c)) and c not in "\t"]
        for c in special_chars:
            special_chars_cnt[c] = special_chars_cnt.get(c, 0) + 1
        rejected += 1

partpipe_close()
print(f" `- waiting for encoder threads to finish writing part files.")
partpipe_close(0)

print(f" `- consolidating {len(datafile_parts)} part files into one large output file.")
with datafile.open("wb") as f:
    for fn in datafile_parts:
        f.write(open(fn, "rb").read())
        os.remove(fn)

print(f"Rejected {rejected} / {total} items (={100*rejected//total}%) containing non-ASCII chars.\n")

if "-s" in opts:
    print("Frequency of non-ASCII chars:")
    for cnt, ch in sorted((-cnt,ch) for ch,cnt in special_chars_cnt.items()):
        print(f"  {ch}\t\\u{hex(ord(ch))[2:]}\t{-cnt}")

print(f"\nFinal Meta:")
for key in sorted(meta.keys()):
    if key in ("itos", "stoi"):
        print(f"    {key:<10} {repr(meta[key])[:60]} ....")
    else:
        print(f"    {key:<10} {repr(meta[key])}")

with metafile.open('wb') as f:
    pickle.dump(meta, f)

print(f"\n{'='*70}\n")
