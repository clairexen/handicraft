#!/usr/bin/env python3

import os
import numpy as np
import xbitpermsyn

if True:
    # pre-generated permutation (for better reproducibility)
    perm = [
        (0, 23), (1, 19), (2, 28), (3, 8), (4, 4), (5, 1), (6, 31), (7, 22), (8, 26), (9, 0),
        (10, 10), (11, 15), (12, 2), (13, 7), (14, 21), (15, 27), (16, 16), (17, 12), (18, 5),
        (19, 24), (20, 25), (21, 20), (22, 30), (23, 29), (24, 6), (25, 13), (26, 14), (27, 11),
        (28, 18), (29, 9), (30, 17), (31, 3)
    ]
else:
    bit_indices = np.arange(32)
    perm = list(zip(bit_indices, np.random.permutation(bit_indices)))
    # perm = perm[0:16]
    print(perm)

os.makedirs("make_randperm", exist_ok=True)
syn = xbitpermsyn.Syn(32, perm, tempdir="make_randperm")

syn.addcommand(rori=True, grevi=True)
syn.addcommand(shuffle=True)
syn.addcommand(shuffle=True)
syn.addcommand(shuffle=True)
syn.addcommand(shuffle=True)
syn.addcommand(shuffle=True)
syn.addcommand(shuffle=True)
syn.addcommand(rori=True, grevi=True)

syn.solve()

print("Solving time: %d:%02d" % (syn.seconds // 60, syn.seconds % 60))
print("----")

if syn.sat:
    for cmd in syn.cprog:
        print(cmd)

    print("----")

    for cmd in syn.aprog:
        print(cmd)

else:
    print("unsat")
