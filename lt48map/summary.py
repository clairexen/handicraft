#!/usr/bin/env python3

for design in [
    "picorv32",
    "nerv",
#   "test",
]:
    luts = 0

    mux_cells = 0
    cc2_cells = 0
    cc3_cells = 0
    cc4_cells = 0
    cc5_cells = 0

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
            if line[0] == "CC_MUX": mux_cells += int(line[1])
            elif line[0].startswith("CC2_"): cc2_cells += int(line[1])
            elif line[0].startswith("CC3_"): cc3_cells += int(line[1])
            elif line[0].startswith("CC4_"): cc4_cells += int(line[1])
            elif line[0].startswith("CC5_"): cc5_cells += int(line[1])

    cpe_min = mux_cells//2 + cc3_cells//2 + cc4_cells//2 + cc5_cells//2
    cpe_max = cpe_min + cc2_cells//2 +  cc5_cells//4
    print("%-8s | %4d 2x4-LUTs %6d-%4d CPEs | %5d MUX %5d CC2 %5d CC3 %5d CC4 %5d CC5" %
            (design, luts//2, cpe_min, cpe_max, mux_cells, cc2_cells, cc3_cells, cc4_cells, cc5_cells))
