import datasets, pickle, pcre2
import json, numpy, sys, os
import config, tokens, utils
from pathlib import Path
from subprocess import Popen, PIPE

dataset_name = "openwebtext"

shortend_meta_fields = set("itos stoi kwtoi tokens".split())

opts, args = utils.opts_args()

if len(args) == 0:
    args.append(config.cfg.name)

if len(args) == 1:
    os.system(f"set -x; python3 '{sys.argv[0]}' {' '.join(opts)} {args[0]} test")
    os.system(f"set -x; python3 '{sys.argv[0]}' {' '.join(opts)} {args[0]} train")
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
    if key in shortend_meta_fields:
        print(f"  {key:<15} {repr(meta[key])[:60]} ....")
    else:
        print(f"  {key:<15} {repr(meta[key])}")

if split == "train":
    dataset = datasets["train"]
    datafile = datapath.joinpath(f"{cfg_name}.train.{lex.binext}")
else:
    dataset = datasets["test"]
    datafile = datapath.joinpath(f"{cfg_name}.test.{lex.binext}")

# ======================================================================

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
        # datafile_partpipes.append(Popen(["/bin/sh", "-c", f"python3 tokens.py -e '{datafile_parts[-1]}'"], stdin=PIPE))
        datafile_partpipes.append(Popen(["./fastenc/fastenc", datafile_parts[-1]], stdin=PIPE))
        datafile_partidx += 1
        datafile_bytes = 0
    datafile_partpipes[-1].stdin.write(bytes(t, "ascii"))
    datafile_bytes += len(t)

# ======================================================================

replace_special_table = {
    "\u0081": '', "\u00b4": "'", "\u2014": '-', "\u201c": '"', "\uff01": '!', "\u00a9": '(C)',
    "\u00ad": '', "\u2011": '-', "\u2015": '-', "\u201d": '"', "\uff08": '(', "\u00ae": '(R)',
    "\u200b": '', "\u2013": '-', "\u2019": "'", "\u201f": '"', "\uff09": ')', "\u2026": '...',
    "\t": "    ", "\uff0d": '"', "\u2018": "'", "\u2212": '-', "\uff0c": ',', "\u00bb": '"',
    "\r":     '', "\u2032": "'", "\u2033": '"', "\u00d7": '*', "\uff0e": '.', "\u00ab": '"',
}

for line in json.load(open("emoji_codes.json")):
    code, tag = line.split(" -- ")[1].split()
    emoji = "".join(chr(int(s,16)) for s in code.split("-"))
    replace_special_table[emoji] = tag

re_special_pat = "|".join(replace_special_table.keys()).replace("*", "\\*")
re_special = pcre2.compile(re_special_pat, jit=True)

def convert_special_chars(t):
    return re_special.sub(lambda m: replace_special_table[m[0]], t)

def escape_unicode(t):
    out = []
    for c in t:
        idx = ord(c)
        if 31 < idx < 127:
            out.append(c)
        elif idx <= 127:
            out.append(f"\\x{idx:02x}")
        elif idx <= 0xffff:
            out.append(f"\\u{idx:04x}")
        else:
            out.append(f"\\U{idx:04x}")
    return "".join(out)

# ======================================================================

total = 0
rejected = 0
special_chars_cnt = dict()
print(f"\nWriting {datafile} ...")
while total < len(dataset) and total - rejected < limit:
    t = convert_special_chars(dataset[total]["text"]); total += 1
    special_chars = [c for c in t if (ord(c) < 32 or 127 <= ord(c)) and c != "\n"]
    if len(special_chars) < 20 and 100*len(special_chars) < len(t):
        if special_chars:
            t = escape_unicode(t)
        t = f"REM '''\n{t}\n'''\n\x00"
        partpipe_write(t)
    else:
        for c in special_chars:
            special_chars_cnt[c] = special_chars_cnt.get(c, 0) + 1
        with datapath.joinpath(f"rejected.txt").open("a") as rej_f:
            rej_f.write("Rejected bc. of the following non-ASCII chars:\n")
            rej_f.write(repr(special_chars) + "\n\n" + t + "\n\n" + "# " + "="*70 + "\n")
        rejected += 1

partpipe_close()
print(f" `- waiting for encoder threads to finish writing part files.")
partpipe_close(0)

# ======================================================================

print(f" `- consolidating {len(datafile_parts)} part files into one large output file.")
with datafile.open("wb") as f:
    for fn in datafile_parts:
        f.write(open(fn, "rb").read())
        os.remove(fn)

print(f"Rejected {rejected} / {total} items (={100*rejected//total}%) containing non-ASCII chars.\n")

if "-s" in opts:
    print("Frequency of non-ASCII chars:")
    for cnt, ch in sorted((-cnt,ch) for ch,cnt in special_chars_cnt.items()):
        if ord(ch) <= 127:
            print(f"  {ch}\t\\x{ord(ch):02x}       {-cnt}")
        elif ord(ch) <= 0xFFFF:
            print(f"  {ch}\t\\u{ord(ch):04x}     {-cnt}")
        else:
            print(f"  {ch}\t\\U{ord(ch):08x} {-cnt}")

if "special_chars" not in meta:
    meta["special_chars"] = dict()
for idx,(cnt,ch) in enumerate(sorted((-cnt,ch) for ch,cnt in special_chars_cnt.items())):
    if idx <= 100 or cnt <= -100:
        meta["special_chars"][ch] = meta["special_chars"].get(ch, 0) - cnt

print(f"\nFinal Meta:")
for key in sorted(meta.keys()):
    if key in shortend_meta_fields:
        print(f"  {key:<15} {repr(meta[key])[:60]} ....")
    else:
        print(f"  {key:<15} {repr(meta[key])}")

with metafile.open('wb') as f:
    pickle.dump(meta, f)

print(f"\n{'='*70}\n")
