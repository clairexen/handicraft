#!/usr/bin/env python3

from sys import argv, exit

expectations = list()
cursor = 0

with open(argv[1], "r") as f:
    for line in f:
        line = line.split()
        if len(line) >= 3 and line[0] == "#" and line[1].startswith("expect-io"):
            expectations.append([line[1].lstrip("expect-"), line[2]])

with open(argv[2], "r") as f:
    for line in f:
        line = line.split()
        if line[0] == "WARNING:" and "readmemh" in line[2]:
            continue
        assert(line == expectations[cursor])
        cursor += 1

assert(cursor == len(expectations))
exit(0)

