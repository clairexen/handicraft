#!/usr/bin/env python3

import re
import fileinput

verbose = False

rel_branch_insns = set("""
    beq bne blt bge bltu bgeu
    jal c.j c.jal c.beqz c.bnez
""".split())

reg_branch_insns = set("""
    jalr c.jalr c.jr
""".split())

db_insncount = 0
db_bytecount = 0
db_patterncount = dict()
db_patternsize = dict()
db_valuecount = dict()

def printBlock(insns):
    for i, insn in enumerate(insns):
        print("%3d %8s %-8s %s %s" % (i, insn[0], insn[1], insn[2], ",".join(insn[3:])))
    
def processSimpleBlock(insns, label):
    global db_insncount, db_bytecount

    foundPatterns = []
    db_insncount += len(insns)

    st_li = dict()
    st_muli = dict()

    found_muli = dict()
    found_muliadd = dict()

    for cursor in range(len(insns)):
        insn = insns[cursor]
        ilen = len(insn[1]) // 2
        db_bytecount += ilen

        is_li = False
        is_muli = False

        if insn[2] in ("addi", "addiw") and insn[4] == "x0":
            if insn[3] != "x0":
                st_li[insn[3]] = (int(insn[5]), ilen)
                is_li = True

        if insn[2] == "c.li":
            if insn[3] != "x0":
                st_li[insn[3]] = (int(insn[4]), ilen)
                is_li = True

        if insn[2] == "c.mv":
            if insn[4] in st_li:
                st_li[insn[3]] = (st_li[insn[4]], ilen)
                is_li = True

        for arg in [4, 5]:
            if insn[2] in ("mul", "mulw") and insn[arg] in st_li:
                st_muli[insn[3]] = (st_li[insn[arg]][0], cursor, ilen+st_li[insn[arg]][1])
                found_muli[cursor] = (st_li[insn[arg]][0], ilen+st_li[insn[arg]][1])
                is_muli = True

        for arg in [4, 5]:
            if insn[2] in ("add", "addw") and insn[arg] in st_muli:
                c = st_muli[insn[arg]][1]
                if c in found_muli:
                    found_muliadd[cursor] = (found_muli[c][0], ilen+found_muli[c][1])
                    del found_muli[c]

        for arg in [3, 4]:
            if insn[2] in ("c.add", "c.addw") and insn[arg] in st_muli:
                c = st_muli[insn[arg]][1]
                if c in found_muli:
                    found_muliadd[cursor] = (found_muli[c][0], ilen+found_muli[c][1])
                    del found_muli[c]

        if len(insn) > 3:
            if not is_li and insn[3] in st_li:
                del st_li[insn[3]]
            for i in range(4 if is_muli else 3, len(insn)):
                if is_muli and insn[i] == insn[3]:
                    continue
                if insn[i] in st_muli:
                    del st_muli[insn[i]]

    for c, (v, l) in found_muli.items():
        foundPatterns.append((insns[c][0], "muli", v, l))

    for c, (v, l) in found_muliadd.items():
        foundPatterns.append((insns[c][0], "muliadd", v, l))

    st_add = dict()
    st_addi = dict()

    for cursor in range(len(insns)):
        insn = insns[cursor]
        ilen = len(insn[1]) // 2

        is_add = False
        is_addi = False
        rd, rs1, rs2, imm = None, None, None, None

        if insn[2] in ("add", "addw") and "x0" not in insn[3:]:
            is_add = True
            rd = insn[3]
            rs1 = insn[4]
            rs2 = insn[5]

        if insn[2] in ("c.add", "c.addw") and "x0" not in insn[3:]:
            is_add = True
            rd = insn[3]
            rs1 = insn[3]
            rs2 = insn[4]

        if insn[2] in ("addi", "addiw") and "x0" not in insn[3:]:
            is_addi = True
            rd = insn[3]
            rs1 = insn[4]
            imm = int(insn[5])

        if insn[2] in ("c.addi", "c.addiw") and "x0" not in insn[3:]:
            is_addi = True
            rd = insn[3]
            rs1 = insn[3]
            imm = int(insn[4])

        if is_add:
            if rs1 in st_addi and ilen+st_addi[rs1][0] > 4:
                foundPatterns.append((cursor, "addiadd", st_addi[rs1][1], ilen+st_addi[rs1][0]))
            if rs2 in st_addi and ilen+st_addi[rs2][0] > 4:
                foundPatterns.append((cursor, "addiadd", st_addi[rs2][1], ilen+st_addi[rs2][0]))

        if is_addi:
            if rs1 in st_add and ilen+st_add[rs1] > 4:
                foundPatterns.append((cursor, "addiadd", imm, ilen+st_add[rs1]))

        if len(insn) > 3:
            for reg in insn[3:]:
                if reg in st_add:
                    del st_add[reg]
                if reg in st_addi:
                    del st_addi[reg]

        if is_add:
            st_add[insn[3]] = ilen

        if is_addi:
            st_addi[insn[3]] = (ilen, imm)

    if len(foundPatterns):
        if verbose:
            print("==== %s ====" % (label,))
        for a, p, v, l in foundPatterns:
            if p not in db_patterncount:
                db_patterncount[p] = 0
                db_patternsize[p] = 0
                db_valuecount[p] = dict()
            if v not in db_valuecount[p]:
                db_valuecount[p][v] = 0
            db_patterncount[p] += 1
            db_patternsize[p] += l
            db_valuecount[p][v] += 1
            print("PATTERN: %s with length %d at %s with value %s" % (p, l, a, v))
        if verbose:
            printBlock(insns)

def processBlock(insns, label):
    blockStart = set()
    nextBlock = True
    for insn in insns:
        if nextBlock:
            blockStart.add(insn[0])
            nextBlock = False
        if insn[2] in rel_branch_insns:
            blockStart.add(insn[-1])
            nextBlock = True
        if insn[2] in reg_branch_insns:
            nextBlock = True

    block = []
    for insn in insns:
        if insn[0] in blockStart:
            if len(block):
                processSimpleBlock(block, label)
            block = []
        block.append(insn)
    if len(block):
        processSimpleBlock(block, label)

def parse():
    label = ""
    insns = []
    for line in fileinput.input():
        m = re.match(r"^\s+([0-9a-fA-F]+):\s+([0-9a-fA-F]+)\s+(\S+)\s*(\S*)", line)
        if m:
            insns.append(tuple([m.group(1), m.group(2), m.group(3)] +
                                m.group(4).replace("(", ",").replace(")", "").split(",")))
        else:
            if len(insns):
                processBlock(insns, label)
            insns = []
            label = line.strip()
            continue
    if len(insns):
        processBlock(insns, label)

parse()

print()
print("Instruction count: %d (%d bytes, avg %.2f bytes/insn)" % (db_insncount, db_bytecount, db_bytecount/db_insncount))
print()

outlines = []
for p in db_patterncount:
    cnt9bit = 0
    for v, c in db_valuecount[p].items():
        if -128 <= v < 128:
            cnt9bit += c
    outlines.append((db_patterncount[p], p, "%20s %8d 12bit=%.4f%% 9bit=%.4f%% avgbytes=%.2f" %
            (p, db_patterncount[p], 100*db_patterncount[p]/db_insncount, 100*cnt9bit/db_insncount, db_patternsize[p]/db_patterncount[p])))

for _, _, line in reversed(sorted(outlines)):
    print(line)
print()

for _, p, _ in reversed(sorted(outlines)):
    print("Values for %s patterns:" % p)
    for c, v in reversed(sorted([(c, v) for v, c in db_valuecount[p].items()])):
        print("%20s %8d" % (v, c))
    print()
