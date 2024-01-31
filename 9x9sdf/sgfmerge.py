#!/usr/bin/env python3

from glob import glob
from collections import defaultdict
from sgfutils import *

counted_positions = defaultdict(int)
relevant_pos_count = 0

for filename in glob("*.sgf"):
    print(filename)
    sgfdata = sgfread(filename)
    # print(sgfprettystr(sgfdata))

    positions = set()
    sgfpositions(sgfdata, positions)

    for pos in positions:
        for ppos in sgfpermpos(pos):
            counted_positions[ppos] += 1

for pos, cnt in counted_positions.items():
    if cnt < 5:
        continue
    print("+---------+")
    for line in pos:
        print("|%s|" % line)
    relevant_pos_count += 1
print("+---------+")

print()
print("Found %d relevant positions.\n" % relevant_pos_count)

