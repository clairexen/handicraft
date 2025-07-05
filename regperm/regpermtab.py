
from collections import defaultdict

Id = "543210"
Zip = "432105"
Refl = "012345"
Shfl = "524130"

def applyOp(dat, op, cnt=1):
    for i in range(cnt):
        dat = "".join(dat[5-int(op[i])] for i in range(6))
    return dat

match 3:
    case 0: MagicOps = []
    case 1: MagicOps = ['043251', '143205']
    case 2: MagicOps = ['053214', '403215']
    case 3: MagicOps = ['143250', '503214']

print()
print("Basic Ops:")
ops = set()
for k in [Id, Refl]:
    for j in range(4):
        for i in range(6):
            ops.add(applyOp(applyOp(k, Zip, i), Shfl, j))
        for m in MagicOps:
            ops.add(applyOp(applyOp(k, m), Shfl, j))
print(f"{len(ops)=}, {sorted(ops)=}")

print()
print("MSB bit pair:")
table = {f"{i}{j}": set() for i in range(6) for j in range(6) if i != j}
for op in ops:
    table[op[:2]].add(op)
for k,v in sorted(table.items()):
    print(k, v)

print()
print("LSB bit pair:")
table = {f"{i}{j}": set() for i in range(6) for j in range(6) if i != j}
for op in ops:
    table[op[4:6]].add(op)
for k,v in sorted(table.items()):
    print(k, v)

print()
print("MSB bit tripple:")
table = {f"{i}{j}{k}": set() for i in range(6) for j in range(6) for k in range(6) if i < j < k}
for op in ops:
    table["".join(sorted(op[:3]))].add(op)
for k,v in sorted(table.items()):
    print(k, v)

print()
print("LSB bit tripple:")
table = {f"{i}{j}{k}": set() for i in range(6) for j in range(6) for k in range(6) if i < j < k}
for op in ops:
    table["".join(sorted(op[3:6]))].add(op)
for k,v in sorted(table.items()):
    print(k, v)

print()
print("Two Ops:")
opops = defaultdict(float)
for op in [applyOp(a, b) for a in ops for b in ops]:
    opops[op] += 1
print(len(opops), ".... Wohho!  \\o/" if len(opops) == 720 else ".... Ohno!  /o\\")

opopops = defaultdict(float)
for cover, op in [(opops[a], applyOp(a, b)) for a in opops for b in ops]:
    opopops[op] += cover**0.5
uncoveredOps = set(op for op in opopops.keys() if op not in opops)
print(f"{len(uncoveredOps)=}, {sorted(uncoveredOps)=}")

print()
print("Three Ops:")
exoticOps = set(op for op, cover in opopops.items() if cover == min(opopops.values()))
print(f"{len(opopops)=}, {min(opopops.values())=}, {len(exoticOps)=}, {sorted(exoticOps)=}")

print()
print("Testing Candidates..")
best = set()
maxScore = 0
for op in uncoveredOps:
    spread = set()
    for k in [Id, Refl]:
        for j in range(4):
            o = applyOp(applyOp(k, op), Shfl, j)
            for a in ops:
                spread.add(applyOp(a, o))
                spread.add(applyOp(o, a))
    score = 0
    for a in spread:
        if a in uncoveredOps: score += 1
    if maxScore < score:
        best = set()
        maxScore = score
    if maxScore == score:
        best.add(op)
    # print(f"{op=} {score=} /{len(spread)}")
print(f"{maxScore=}, {len(best)=}, {sorted(best)=}")

print()
print("Inspecting Candidates..")
for op in best:
    loop = set()
    for i in range(1, 10):
        if (o := applyOp(Id, op, i)) == Id: break
        loop.add(o)
    print(f"{op=} {len(loop)=} {sorted(loop)=}")

