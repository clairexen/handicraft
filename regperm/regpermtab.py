
from collections import defaultdict

Id = "543210"
Zip = "432105"
Refl = "012345"
Shfl = "524130"

MagicOps = ["012543", "512304", "253401", "201435", "024315"]

def applyOp(dat, op, cnt=1):
    for i in range(cnt):
        dat = "".join(dat[5-int(op[i])] for i in range(6))
    return dat

print()
print("Basic Ops:")
ops = set()
for k in range(2):
    for i in range(6):
        for j in range(4):
            ops.add(applyOp(applyOp(Refl if k else Id, Zip, i), Shfl, j))
    for i in range(len(MagicOps)):
        if False:
            for j in range(4):
                ops.add(applyOp(applyOp(Refl if k else Id, MagicOps[i]), Shfl, j))
        else:
            ops.add(applyOp(Refl if k else Id, MagicOps[i]))
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
print("Two Ops:")
opops = defaultdict(int)
for op in [applyOp(a, b) for a in ops for b in ops]:
    opops[op] += 1
print(len(opops))

print()
print("Three Ops:")
opopops = defaultdict(int)
for cost, op in [(opops[a], applyOp(a, b)) for a in opops for b in ops]:
    opopops[op] += cost
exoticOps = sorted(op for op, cost in opopops.items() if cost == min(opopops.values()))
print(f"{len(opopops)=}, {min(opopops.values())=}, {len(exoticOps)=}, {exoticOps=}")




