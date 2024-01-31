#!/usr/bin/env python3

import os
import xbitpermsyn

if False:
    # J-type RISC-V immediate (output aligned to MSB end)
    perm = [ (31, 11+20), (20, 11+11) ]
    perm += [(20+i, 11+i) for i in range(1, 11)]
    perm += [(12+i-12, 11+i) for i in range(12, 20)]
else:
    # Immediate bits 11:1 in J-type RISC-V
    perm = [ (20, 11+11) ]
    perm += [(20+i, 11+i) for i in range(1, 11)]

os.makedirs("make_riscv_j", exist_ok=True)
syn = xbitpermsyn.Syn(32, perm, tempdir="make_riscv_j", order_mode=True)

syn.addcommand(rori=True, grevi=True, gzip=True)

syn.solve()
print("Solving time: %d:%02d" % (syn.seconds // 60, syn.seconds % 60))
print("----")

if syn.sat:
    for cmd in syn.cprog:
        print(cmd)

    print("t0 = 0x%08x;" % syn.mask)
    print("a0 = bext(a0, t0);")
    print("a0 = sll(a0, 12);")
    print("a0 = sra(a0, 11);")

    print("----")

    for cmd in syn.aprog:
        print(cmd)

    print("li t0, 0x%08x" % syn.mask)
    print("bext a0, a0, t0")
    print("c.slli a0, 12")
    print("c.srai a0, 11")

else:
    print("unsat")
