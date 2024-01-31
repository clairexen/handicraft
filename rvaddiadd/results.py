#!/usr/bin/env python3

import re
import sys
import numpy as np

insncnt = 0
insnbytes = []
db_types = set()
db_counts = dict()
db_values = dict()

for filename in sys.argv[1:]:
    with open(filename, "r") as f:
        db_counts[filename] = dict()
        db_values[filename] = dict()
        valblock = None
        patblock = False
        for line in f:
            line = line.rstrip()
            if line == "":
                continue
            m = re.match(r"^Instruction count: ([0-9]+) \([0-9]+ bytes, avg ([0-9.]+) bytes/insn\)$", line)
            if m:
                insncnt += int(m.group(1))
                insnbytes.append(float(m.group(2)))
                patblock = True
                continue
            m = re.match(r"^Values for ([a-z]+) patterns:$", line)
            if m:
                valblock = m.group(1)
                patblock = False
                continue
            if valblock is None:
                if patblock:
                    m = re.match(r"^ +([a-z]+) +([0-9]+) +12bit=([0-9.]+)% +9bit=([0-9.]+)% avgbytes=([0-9.]+)$", line)
                    if m:
                        t = m.group(1)
                        p12 = float(m.group(3))
                        p9 = float(m.group(4))
                        b = float(m.group(5))
                        db_types.add(t)
                        db_counts[filename][t] = (p12, p9, b)
                        continue
                    else:
                        patblock = False
            else:
                m = re.match(r"^ +(-?[0-9]+) +([0-9]+)$", line)
                if m:
                    v = int(m.group(1))
                    c = int(m.group(2))
                    if valblock not in db_values[filename]:
                        db_values[filename][valblock] = dict()
                    if v not in db_values[filename][valblock]:
                        db_values[filename][valblock][v] = 0
                    db_values[filename][valblock][v] += c
                    continue

def minmaxmean(vals, unit=""):
    return "MIN=%.3f%s MEDIAN=%.3f%s MEAN=%.3f%s STDDEV=%.3f%s MAX=%.3f%s" % \
            (np.min(vals), unit, np.median(vals), unit, np.mean(vals), unit, np.std(vals), unit, np.max(vals), unit)

print()
print("Package count: %s" % (len(sys.argv)-1))
print("Instruction count: %s" % insncnt)
print()
print("Bytes/insn: %s" % minmaxmean(insnbytes))
print()

for pname, pidx, unit in [("12-bit", 0, "%"), ("9-bit", 1, "%"), ("bytes", 2, "")]:
    for t in db_types:
        vals = []
        for n in db_counts:
            if t in db_counts[n]:
                vals.append(db_counts[n][t][pidx])
            elif pname != "bytes":
                vals.append(0)
        print("%10s %8s %s" % (t, pname, minmaxmean(vals, unit)))
    print()

print("%10s" % "", end="")
for N in [4, 5, 6, 7, 8, 9, 10, 11, 12]:
    print(" %2d-bits" % N, end="")
print()

for t in db_types:
    if t == "DOUBLE_C":
        continue
    cnt = 0
    vals = dict()
    for n in db_values:
        if t in db_values[n]:
            for v, c in db_values[n][t].items():
                cnt += c
                if -8 <= v < 8:
                    vals.setdefault(4, 0)
                    vals[4] += c
                elif -16 <= v < 16:
                    vals.setdefault(5, 0)
                    vals[5] += c
                elif -32 <= v < 32:
                    vals.setdefault(6, 0)
                    vals[6] += c
                elif -64 <= v < 64:
                    vals.setdefault(7, 0)
                    vals[7] += c
                elif -128 <= v < 128:
                    vals.setdefault(8, 0)
                    vals[8] += c
                elif -256 <= v < 256:
                    vals.setdefault(9, 0)
                    vals[9] += c
                elif -512 <= v < 512:
                    vals.setdefault(10, 0)
                    vals[10] += c
                elif -1024 <= v < 1024:
                    vals.setdefault(11, 0)
                    vals[11] += c
                elif -2048 <= v < 2048:
                    vals.setdefault(12, 0)
                    vals[12] += c

    print("%10s" % (t,), end="")
    v = 0
    for N in [4, 5, 6, 7, 8, 9, 10, 11, 12]:
        v += 100 * vals[N] / cnt
        print("%7.2f%%" % v, end="")
    print()
print()
