
###################################################################
# Fundamental Ops

Id = "543210"
Zip = "432105"
Refl = "012345"
Shfl = "524130"

# the chosen "magic ops pair"
MagicOps = {'543201', '543120'}


###################################################################
# Fundamental Op Groups

def applyOp(dat, op, cnt=1):
    if isinstance(dat, str):
        for i in range(cnt):
            dat = "".join(dat[5-int(op[i])] for i in range(6))
        return dat
    if isinstance(dat, set):
        return {applyOp(d, op) for d in dat}
    assert False

ReflOps = {Id, Refl}
ZipOps = {applyOp(Id, Zip, i) for i in range(6)}
ZipOps5 = {applyOp(Id, Zip, i) for i in range(6) if i != 3}
ZipMagicOps = ZipOps | MagicOps
ShflOps = {applyOp(Id, Shfl, i) for i in range(4)}
ShflOps3 = {applyOp(Id, Shfl, i) for i in range(4) if i != 2}


###################################################################
# The "Regular Permutations Engine"

def applyOps(dats, ops):
    return {applyOp(dat, op) for dat in dats for op in ops}

OpChain = []
ValChain = [{Id}]

def addStage(ops):
    OpChain.append(ops)
    ValChain.append(applyOps(ValChain[-1], ops))
    print(f"Chain Link #{len(OpChain)}: {len(ops)} ops, {len(ValChain[-1])} outputs")

addStage(ReflOps)
addStage(ZipOps)
addStage(ShflOps)
addStage(ZipMagicOps)
addStage(ShflOps)
assert(len(ValChain[-1]) == 720)


###################################################################
# R-Type Ops only have access to the first three stages

rTypeOps = ValChain[3]
#rTypeOps = applyOps(applyOps(ReflOps, ZipOps), ShflOps3)
#rTypeOps = applyOps(applyOps(ReflOps, ZipOps5), ShflOps)

def evalRTypePermOps(ops):
    outTable = [[],[],[]]
    outTable[0].append("All 30 ordered MSB pairs:")
    outTable[1].append("All 30 ordered LSB pairs:")
    table = {f"{i}{j}": [set(),set()] for i in range(6) for j in range(6) if i != j}
    for op in ops:
        table[op[:2]][0].add(op)
        table[op[4:]][1].add(op)
    for k,(v0,v1) in sorted(table.items()):
        outTable[0].append(f"{k}-XX-XX: {' '.join(sorted(v0))}")
        outTable[1].append(f"XX-XX-{k}: {' '.join(sorted(v1))}")

    outTable[2].append("All 20 unordered MSB-LSB tripples:")
    table = {f"{i}{j}{k}": [set(), ""] for i in range(6) for j in range(6) for k in range(6) if i < j < k}
    for op in ops:
        table["".join(sorted(op[:3]))][0].add(op)
        table["".join(sorted(op[:3]))][1] = "".join(sorted(op[3:]))
    for k0,(v,k1) in sorted(table.items()):
        outTable[2].append(f"{k0}-{k1}: {' '.join(sorted(v))}")

    while len(outTable[2]) < len(outTable[0]):
        outTable[2].append("")

    for i in range(len(outTable[0])):
        if i == 0:
            print(f"    | {outTable[0][i]:25} | {outTable[1][i]:25} | {outTable[2][i]}")
        else:
            print(f"{i:2}. | {outTable[0][i]:25} | {outTable[1][i]:25} | {outTable[2][i]}")

print()
print(f"Number of Ops for R-Type INSN: {len(rTypeOps)}")
evalRTypePermOps(rTypeOps)


###################################################################
# Find Magic Op Pairs

print()

ops1 = ValChain[3]
ops2 = applyOps(ops1, ops1)
ops3 = applyOps(ops2, ops1)
assert len(ops2) < 720 and len(ops3) == 720
hardOps = ops3 - ops2

print()
print(f"{len(ops1)=} {len(ops2)=} {len(ops3)=} {len(hardOps)=}")

# find candidates for first magic op. it can be any of the 720,
# but must cover at least half of the hardOps.
sel_candidates = dict()
all_candidates = dict()
for op in ops3:
    spread = applyOps(applyOp(ops1, op), ShflOps) & hardOps
    if len(spread) >= len(hardOps) // 2:
        sel_candidates[op] = spread
    all_candidates[op] = spread
print(f"{len(sel_candidates)=} {sorted(sel_candidates)=}")

print()
magic_pairs = set()
for a, a_spread in sorted(sel_candidates.items()):
    magic_partners = set()
    for b, b_spread in all_candidates.items():
        if len(a_spread) + len(b_spread) < len(hardOps):
            continue
        spread = a_spread | b_spread
        if len(spread) >= len(hardOps):
            assert len(spread) == len(hardOps)
            magic_pairs.add(tuple(reversed(sorted([a, b]))))
            magic_partners.add(b)
    assert magic_partners
    if magic_partners and a[:3] == "543":
        print(f"{a=} {len(magic_partners)=} {sorted(magic_partners)[-5:]=}")
print()
print(f"{len(magic_pairs)=} {sorted(magic_pairs)[-3:]=}")
