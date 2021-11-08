#!/usr/bin/env python3

for design in ["picorv32", "nerv", "test"]:
    luts = 0
    cc_cells = 0
    cc_gates = 0
    with open(design + "_l4.stat") as f:
        for line in f:
            if not line.startswith("     "):
                continue
            line = line.split()
            if len(line) != 2:
                continue
            if line[0] == "$lut":
                luts += int(line[1])
    with open(design + "_lt4.stat") as f:
        for line in f:
            if not line.startswith("     "):
                continue
            line = line.split()
            if len(line) != 2:
                continue
            if line[0].startswith("C3_"): cc_cells += int(line[1])
            if line[0].startswith("C4_"): cc_cells += int(line[1])
            if line[0].startswith("C5_"): cc_cells += int(line[1])
            if line[0] in ["AND", "XOR", "OR"]: cc_gates += int(line[1])
    print("%-12s %4d 4-LUTs %8d-%4d CPEs" % (design, luts, cc_cells//2, (cc_cells+cc_gates)//2))
