#!/usr/bin/env python3

import gzip
import json
import random
import sys

MAX_TRAIN = None # no limit
MAX_BYTES = 2 * 1024  # 2 kB
TEST_SPLIT = 0.11
SEP = "\n\n<|----|>\n\n"

INPUT = "dolma-cccc-filtered-0000.json.gz"
OUT_TRAIN = "cccclc-train.txt.gz"
OUT_TEST = "cccclc-test.txt.gz"

#random.seed()    # nondeterministic
random.seed(1234) # set a seed here if you want reproducibility

def split_center_paragraph(text):
    """
    Split text at the most central '\n\n'.
    If none exists, return None.
    """
    splits = [i for i in range(len(text)) if text.startswith("\n\n", i)]
    if not splits:
        return None
    center = len(text) // 2
    i = min(splits, key=lambda x: abs(x - center))
    return text[:i], text[i + 2:]

def explode_text(text):
    """
    Recursively split text until all chunks are < MAX_BYTES.
    """
    out = [text]
    changed = True

    while changed:
        changed = False
        new_out = []
        for t in out:
            if len(t.encode("utf-8")) <= MAX_BYTES:
                new_out.append(t)
                continue

            res = split_center_paragraph(t)
            if res is None:
                # hard fallback: byte split
                mid = len(t) // 2
                new_out.extend([t[:mid], t[mid:]])
            else:
                new_out.extend(res)
            changed = True
        out = new_out

    return out

test = []
train = []

print(f"Read: {INPUT}")
with gzip.open(INPUT, "rt", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        text = obj.get("text", "")
        if not text:
            continue

        text = text.lower()
        chunks = explode_text(text)

        if random.random() < 0.1:
            for c in chunks:
                test.append(c)
        else:
            for c in chunks:
                train.append(c)

        if MAX_TRAIN is not None and len(train) >= MAX_TRAIN:
            break

print(f"test records:  {len(test)}", file=sys.stderr)
print(f"train records: {len(train)}", file=sys.stderr)

print(f"Shuffe: test records")
random.shuffle(test)

print(f"Write: {OUT_TEST}")
with gzip.open(OUT_TEST, "wt", encoding="utf-8") as f:
    for t in test:
        f.write(t.strip())
        f.write(SEP)

print(f"Shuffe: train records")
random.shuffle(train)

print(f"Write: {OUT_TRAIN}")
with gzip.open(OUT_TRAIN, "wt", encoding="utf-8") as f:
    for t in train:
        f.write(t.strip())
        f.write(SEP)
