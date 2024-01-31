#!/usr/bin/env python3

import fileinput

database = dict()
for line in fileinput.input():
    line = line.split()
    cnt = int(line[0])
    cmd = " ".join(line[1:])
    if cmd not in database:
        database[cmd] = cnt
    else:
        database[cmd] += cnt

for cmd, cnt in sorted(database.items()):
    print("%10d %s" % (cnt, cmd))
