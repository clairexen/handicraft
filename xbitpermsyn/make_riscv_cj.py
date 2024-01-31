#!/usr/bin/env python3

import os
import xbitpermsyn

if False:
    # CJ-type RISC-V immediate (output aligned to MSB end)
    perm = [
        (12, 20 + 11),
        (11, 20 +  4),
        (10, 20 +  9),
        ( 9, 20 +  8),
        ( 8, 20 + 10),
        ( 7, 20 +  6),
        ( 6, 20 +  7),
        ( 5, 20 +  3),
        ( 4, 20 +  2),
        ( 3, 20 +  1),
        ( 2, 20 +  5),
    ]
else:
    # Only the problematic bits
    perm = [
      # (12, 11),
        (11,  4),
      # (10,  9),
      # ( 9,  8),
        ( 8, 10),
      # ( 7,  6),
        ( 6,  7),
      # ( 5,  3),
      # ( 4,  2),
      # ( 3,  1),
        ( 2,  5),
    ]

os.makedirs("make_riscv_cj", exist_ok=True)
syn = xbitpermsyn.Syn(32, perm, tempdir="make_riscv_cj", order_mode=True)

syn.addcommand(rori=True, grevi=True, gzip=True)

syn.solve()
print("Solving time: %d:%02d" % (syn.seconds // 60, syn.seconds % 60))
print("----")

if syn.sat:
    for cmd in syn.cprog:
        print(cmd)

    print("t0 = 0x%08x;" % syn.mask)
    print("a0 = bext(a0, t0);")
    print("a0 = sll(a0, 21);")
    print("a0 = sra(a0, 20);")

    print("----")

    for cmd in syn.aprog:
        print(cmd)

    print("li t0, 0x%08x" % syn.mask)
    print("bext a0, a0, t0")
    print("c.slli a0, 21")
    print("c.srai a0, 20")

else:
    print("unsat")
