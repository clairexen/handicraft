#!/usr/bin/env python3

import re

slo_ror = 0
lui_orc = 0
packed2 = 0
zipped2 = 0
packed3 = 0
unhandled = 0

with open("aa64lst.txt", "r") as f:
    for line in f:
        val = int(line.split()[0], 16)

        # slo_ror
        s = format(val, '064b')
        if re.match(r"^(0*1+0*|1*0+1*)$", s):
            slo_ror += 1
            continue

        # lui_orc
        v0 = (val >>  0) & 0xFFFF
        v1 = (val >> 16) & 0xFFFF
        v2 = (val >> 32) & 0xFFFF
        v3 = (val >> 48) & 0xFFFF
        if v0 == v1 == v2 == v3:
            lui_orc += 1
            continue

        # packed and zipped
        v0 = (val >>  0) & 0xFFFFFFFF
        v1 = (val >> 32) & 0xFFFFFFFF
        if v0 == v1:
            s = format(v0, '032b')
            if re.match(r"^.{20}0{12}$", s):
                packed2 += 1 # LUI
                continue
            if re.match(r"^(0{21}|1{21}).{11}$", s):
                packed2 += 1 # ADDI
                continue
            if re.match(r"^0+1+$", s):
                packed2 += 1 # SLO
                continue
            if re.match(r"^1+0+$", s):
                packed2 += 1 # SRO
                continue
            if re.match(r"^0*10*$", s):
                packed2 += 1 # SBSET
                continue

            z = "".join([c+c for c in s])
            if re.match(r"^(0{33}|1{33}).{19}0{12}$", z):
                zipped2 += 1 # LUI
                continue

            print("0x%016x" % val, s, "(2x)")
            packed3 += 1
            continue

        print("0x%016x" % val)
        unhandled += 1

print("slo_ror   %5d" % slo_ror)
print("lui_orc   %5d" % lui_orc)
print("packed2   %5d" % packed2)
print("zipped2   %5d" % zipped2)
print("packed3   %5d" % packed3)
print("unhandled %5d" % unhandled)
print("total     %5d" % (slo_ror+lui_orc+packed2+zipped2+packed3+unhandled))

