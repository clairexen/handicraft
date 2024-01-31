#!/usr/bin/env python3

db = dict()
queue = set()

class Entry:
    def __init__(self, value, parents, oper, depth):
        self.value = value
        self.parents = parents
        self.oper = oper
        self.depth = depth

    def getmax(self):
        v = self.value
        for p in self.parents:
            v = max(v, p.getmax())
        return v

    def getmin(self):
        v = self.value
        for p in self.parents:
            v = min(v, p.getmin())
        return v

    def __str__(self):
        if self.oper == "zero":
            return "0"
        if self.oper == "unit":
            return "x"
        return "%s(%s)" % (self.oper, ", ".join([str(p) for p in self.parents]))

def new_entry(value, parents, oper, cost=1):
    depth = cost
    for p in parents:
        depth += p.depth
    if -100 < value < 200 and (value not in db or db[value].depth > depth):
        db[value] = Entry(value, parents, oper, depth)
        queue.add(value)

new_entry(0, [], "zero", 0)
new_entry(1, [], "unit", 0)

while len(queue):
    print("processing.. len(db)=%d, len(queue)=%d" % (len(db), len(queue)))
    oldqueue = list(queue)
    queue = set()

    for value in oldqueue:
        entry = db[value]

        for v, e in list(db.items()):
            new_entry(value+v, [entry, e], "add")
            new_entry(value-v, [entry, e], "sub")
            new_entry(v-value, [e, entry], "sub")

            for i in range(1, 4):
                new_entry((value << i) + v, [entry, e], "sh%dadd" % i)
                new_entry((v << i) + value, [e, entry], "sh%dadd" % i)
                #new_entry((value << i) - v, [entry, e], "sh%dsub" % i)
                #new_entry((v << i) - value, [e, entry], "sh%dsub" % i)

        for i in range(1, 9):
            new_entry(value << i, [entry], "sh%d" % i)

print("---")
histdata = []
maxtmp = 0
mintmp = 0
for i in range(10):
    for j in range(1, 11):
        e = db[10*i+j]
        d = e.depth
        print("%3d" % d, end="")
        maxtmp = max(maxtmp, e.getmax() - e.value)
        mintmp = min(mintmp, e.getmin())
        while len(histdata) <= d:
            histdata.append(0)
        histdata[d] += 1
    print()

print("---")
for i, j in enumerate(histdata):
    print("%3d .. %2dx" % (i, j))

print("---")
print("maxtmp = v+%d" % maxtmp)
print("mintmp = %d" % mintmp)
print("dbsize = %d" % len(db))
print("score = %d" % sum([i*j for i, j in enumerate(histdata)]))

if False:
    print("---")
    out = ["%4d(%d)" % (v, e.depth) for v, e in sorted(db.items())]
    for i in range((len(out)+9) // 10):
        print(" ".join(out[10*i:10*i+10]))

print("---")
for i in [15,16,17,97,99,186]:
    print("%3d = %s" % (i, db[i]))

