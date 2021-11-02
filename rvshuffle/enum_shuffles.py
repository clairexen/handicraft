#!/usr.bin/env python

N = 6

def shfl(data, ctrl):
    assert len(data) == N
    assert len(ctrl) == N-1

    data = list(data)
    for i in range(len(ctrl)):
        if ctrl[i] == "1":
            data[i], data[i+1] = data[i+1], data[i]

    return "".join(data)

def unshfl(data, ctrl):
    assert len(data) == N
    assert len(ctrl) == N-1

    data = list(data)
    for i in range(len(ctrl)-1, -1, -1):
        if ctrl[i] == "1":
            data[i], data[i+1] = data[i+1], data[i]

    return "".join(data)

def zzip(data):
    return shfl(shfl(data, "1"*(N-1)), "1"*(N-1))

def unzzip(data):
    return unshfl(unshfl(data, "1"*(N-1)), "1"*(N-1))

def zzzip(data):
    return shfl(shfl(shfl(data, "1"*(N-1)), "1"*(N-1)), "1"*(N-1))

def revbin(data):
    return "".join(reversed(data))

def zcurve(data):
    a = list(data[0:-(N//2)])
    b = list(data[-(N//2):])
    if len(a) != len(b): b.append("")
    return "".join([A+B for A, B in zip(a, b)])

def makedb(init, maxiter=None):
    db = dict()
    queue = set()
    db[init] = ()
    queue.add(init)
    while len(queue):
        if maxiter is not None:
            if maxiter == 0: break
            maxiter -= 1
        batch = queue
        queue = set()

        for src in batch:
            for trg, info in [
                (shfl(src, b), "shfl(%d)" % int(b, 2)) for b in shflargs
            ] + [
                (unshfl(src, b), "unshfl(%d)" % int(b, 2)) for b in unshflargs
            ] + [
                (zzip(src), "zzip"),
                (unzzip(src), "unzzip"),
                (revbin(src), "revbin"),
            ] + ([
                (zzzip(src), "zzzip"),
            ] if N == 6 else []):
                if trg not in db:
                    db[trg] = db[src] + (info,)
                    queue.add(trg)
    return db

init = "9876543210"[-N:]
shflargs = [bin(i)[2:].rjust(N-1, "0") for i in range(1, 1<<(N-1))]
unshflargs = [b for b in shflargs if shfl(init, b) != unshfl(init, b)]

db = makedb(init)
hard = set()

print()
print("Total database size: %d" % len(db))
longcnt = 0
maxlen = 0
for key, val in sorted(db.items()):
    maxlen = max(maxlen, len(val))
    if len(val) > 2: hard.add(key)
    if len(val) > N//2: longcnt += 1
    print("  %s  %d  %s" % (key, len(val), val))

print()
print("Number of 'long' shuffles: %d / %d" % (longcnt, len(db)))

summary = [0] * (maxlen+1)
for key, val in sorted(db.items()):
    if len(val) > N//2:
        print("  %s  %d  %s" % (key, len(val), val))
    summary[len(val)] += 1

print()
print("Number of 'hard' shuffles: %d / %d" % (len(hard), len(db)))

scoreboard = dict()

for h in hard:
    for key in makedb(h, N//2-1):
        if key not in scoreboard:
            scoreboard[key] = set()
        scoreboard[key].add(h)

scores = sorted([(-len(val), key, val) for key, val in scoreboard.items()])
for _, k, v in scores[:10]:
    print("  %s  %d  %s" % (k, len(v), db[k]))

print()
print("Summary (%d bits):" % (1<<N))
for a, b in enumerate(summary):
    print("  %3d permutation%c with %d instructions" % (b, "s" if b != 1 else " ", a))
print()
