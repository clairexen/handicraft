#!/usr/bin/env python3

import os
import xbitpermsyn

# B-type RISC-V immediate (output aligned to MSB end)
perm = [ (31, 19+12), (7, 19+11) ]
perm += [(7+i, 19+i) for i in range(1, 5)]
perm += [(25+i-5, 19+i) for i in range(5, 11)]

os.makedirs("make_riscv_b", exist_ok=True)
syn = xbitpermsyn.Syn(32, perm, tempdir="make_riscv_b", order_mode=True)

syn.addcommand(rori=True, grevi=True, gzip=True)
syn.addcommand(rori=True, grevi=True, gzip=True)
syn.addcommand(rori=True, grevi=True, gzip=True)

syn.solve()

print("Solving time: %d:%02d" % (syn.seconds // 60, syn.seconds % 60))
print("----")

if syn.sat:
    for cmd in syn.cprog:
        print(cmd)

    print("t0 = 0x%08x;" % syn.mask)
    print("a0 = bext(a0, t0);")
    print("a0 = sll(a0, 20);")
    print("a0 = sra(a0, 19);")

    print("----")

    for cmd in syn.aprog:
        print(cmd)

    print("li t0, 0x%08x" % syn.mask)
    print("bext a0, a0, t0")
    print("c.slli a0, 20")
    print("c.srai a0, 19")

else:
    print("unsat")
