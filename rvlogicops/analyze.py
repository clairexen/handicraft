#!/usr/bin/env python3

import re
import sys
from collections import defaultdict

count_insn = 0
count_seteq = 0
count_setne = 0
count_lor = 0
count_lnor = 0
count_sextb = 0
count_sexth = 0
count_zextb = 0
count_zexth = 0

sextb_src = defaultdict(int)
sexth_src = defaultdict(int)
zextb_src = defaultdict(int)
zexth_src = defaultdict(int)

unaries = defaultdict(int)

def get_ext_src(buf, cursor, reg, db):
    i = cursor - 1
    while i >= 0 and i > cursor-10:
        if len(buf[i]) > 2 and buf[i][2] == buf[cursor][3]:
            db[buf[i][1]] += 1
            return buf[i][1]
        i -= 1
    return "?"

def analyze(buf):
    global count_seteq
    global count_setne
    global count_lor
    global count_lnor
    global count_sextb
    global count_sexth
    global count_zextb
    global count_zexth
    cursor=0
    while cursor < len(buf)-1:
        if buf[cursor][1] in ["sub", "c.sub", "subw", "c.subw"]:
            t = buf[cursor][2]
            i = cursor+ 1
            while i < cursor+10 and i < len(buf):
                if buf[i][1] == "sltu" and buf[i][3] == "x0" and buf[i][4] == t:
                    print("seteq", buf[cursor], buf[i])
                    count_seteq += 1
                if buf[i][1] == "sltiu" and buf[i][3] == t and buf[i][4] == "1":
                    print("setne", buf[cursor], buf[i])
                    count_setne += 1
                i += 1

        if buf[cursor][1] in ["or", "c.or"]:
            t = buf[cursor][2]
            i = cursor+ 1
            while i < cursor+10 and i < len(buf):
                if buf[i][1] == "sltu" and buf[i][3] == "x0" and buf[i][4] == t:
                    print("lor", buf[cursor], buf[i])
                    count_lor += 1
                if buf[i][1] == "sltiu" and buf[i][3] == t and buf[i][4] == "1":
                    print("lnor", buf[cursor], buf[i])
                    count_lnor += 1
                i += 1

        if buf[cursor][1] in ["slli", "slliw", "c.slli"]:
            r = buf[cursor][2 if buf[cursor][1] == "c.slli" else 3]
            s = buf[cursor][3 if buf[cursor][1] == "c.slli" else 4]
            if s in ["0x38", "0x30", "0x18", "0x10"]:
                w = "w" if s in ["0x18", "0x10"] else ""
                t = buf[cursor][2]
                i = cursor+ 1
                while i < cursor+10 and i < len(buf):
                    if (buf[i][1] == ("srai"+w) and buf[i][3] == t and buf[i][4] == s) or \
                            (buf[i][1] == ("c.srai"+w) and buf[i][2] == t and buf[i][3] == s):
                        if s in ["0x38", "0x18"]:
                            print("sextb", buf[cursor], buf[i], get_ext_src(buf, cursor, r, sextb_src))
                            count_sextb += 1
                        else:
                            print("sexth", buf[cursor], buf[i], get_ext_src(buf, cursor, r, sexth_src))
                            count_sexth += 1
                    if (buf[i][1] == ("srli"+w) and buf[i][3] == t and buf[i][4] == s) or \
                            (buf[i][1] == ("c.srli"+w) and buf[i][2] == t and buf[i][3] == s):
                        if s in ["0x38", "0x18"]:
                            print("zextb", buf[cursor], buf[i], get_ext_src(buf, cursor, r, zextb_src))
                            count_zextb += 1
                        else:
                            print("zexth", buf[cursor], buf[i], get_ext_src(buf, cursor, r, zexth_src))
                            count_zexth += 1
                    i += 1

        if buf[cursor][1] == "andi" and buf[cursor][4] == "255":
            print("zextb", buf[cursor], get_ext_src(buf, cursor, buf[cursor][3], zextb_src))
            count_zextb += 1

        if not buf[cursor][1].startswith("c.") and buf[cursor][1] not in ("fld", "lw", "flw", "ld", "fsd", \
                "sq", "sw", "fsw", "sd", "addi", "jal", "lbu", "auipc", "sb", "lwu", "lhu", "slli", "srli"):
            reg_cnt = 0
            alt_args = [buf[cursor][1]]
            for arg in buf[cursor][2:]:
                if arg.startswith("x") or arg.startswith("f"):
                    alt_args.append(arg[0] + "?")
                    reg_cnt += 1
                elif arg.endswith(")") and "(x" in arg:
                    alt_args.append(re.sub("x[0-9]+", "x?", arg))
                    reg_cnt += 1
                else:
                    alt_args.append(arg)
            if reg_cnt < (2 if alt_args[0].startswith("c.") else 3):
                alt_args = " ".join(alt_args)
                unaries[alt_args] += 1

        cursor += 1

insn_re = re.compile(r"^\s+([0-9a-f]+):\s+[0-9a-f]+\s+(\S+)\s+(\S+)")

for fn in sys.argv[1:]:
    with open(fn, "rt") as f:
        buf = list()
        for line in f:
            match = insn_re.match(line)
            if match:
                buf.append([match.group(1), match.group(2)] + match.group(3).split(","))
                count_insn += 1
            else:
                if len(buf):
                    analyze(buf)
                buf = list()
        if len(buf):
            analyze(buf)

def print_srcdb(op, db, total):
    print()
    print("sources for %s:" % op)
    index = 0
    for cnt, src in reversed(sorted([(v, k) for k, v in db.items()])):
        if index > 5:
            print("%10s" % "...")
            break
        print("%10s %6d (%.4f)" % (src, cnt, 100*cnt/total))
        index += 1

print_srcdb("sextb", sextb_src, count_sextb)
print_srcdb("sexth", sexth_src, count_sexth)
print_srcdb("zextb", zextb_src, count_zextb)
print_srcdb("zexth", zexth_src, count_zexth)

print()
print("unaries:")
index = 0
for cnt, insn in reversed(sorted([(v, k) for k, v in unaries.items()])):
    if index > 20:
        print("%20s" % "...")
        break
    print("%20s %6d (%.4f%%)" % (insn, cnt, 100*cnt/count_insn))
    index += 1

print()
print("total %8d" % (count_insn))
print("  seteq %6d (%.4f%%)" % (count_seteq, 100*count_seteq/count_insn))
print("  setne %6d (%.4f%%)" % (count_setne, 100*count_setne/count_insn))
print("  lor   %6d (%.4f%%)" % (count_lor, 100*count_lor/count_insn))
print("  lnor  %6d (%.4f%%)" % (count_lnor, 100*count_lnor/count_insn))
print("  sextb %6d (%.4f%%)" % (count_sextb, 100*count_sextb/count_insn))
print("  sexth %6d (%.4f%%)" % (count_sexth, 100*count_sexth/count_insn))
print("  zextb %6d (%.4f%%)" % (count_zextb, 100*count_zextb/count_insn))
print("  zexth %6d (%.4f%%)" % (count_zexth, 100*count_zexth/count_insn))
