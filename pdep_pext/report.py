#!/usr/bin/env python3

mod_gates = dict()
mod_tran = dict()

with open("pdep_pext.rpt") as f:
    for line in f:
        if line.startswith("==="):
            modname = line.split()[1]
            continue
        if "Number of cells:" in line:
            mod_gates[modname] = int(line.split()[-1])
            mod_tran[modname] = 0
        if "$_NOR_" in line:
            mod_tran[modname] += 4*int(line.split()[-1])
        if "$_NAND_" in line:
            mod_tran[modname] += 4*int(line.split()[-1])
        if "$_NOT_" in line:
            mod_tran[modname] += 2*int(line.split()[-1])

print()
print("ASIC (NAND/NOR/NOT) gate counts:")
print("  Control Circuits        %5d   (%3d%%)" % (mod_gates["pdep_pext"], round(100*mod_gates["pdep_pext"]/mod_gates["design"])))
print("  Butterfly Networks      %5d   (%3d%%)" % (mod_gates["pdep_pext_butterfly"], round(100*mod_gates["pdep_pext_butterfly"]/mod_gates["design"])))
print("  Parallel Prefix Count   %5d   (%3d%%)" % (mod_gates["pdep_pext_ppc"], round(100*mod_gates["pdep_pext_ppc"]/mod_gates["design"])))
print("  LROTC of Zero Units     %5d   (%3d%%)" % (mod_gates["pdep_pext_decoder"], round(100*mod_gates["pdep_pext_decoder"]/mod_gates["design"])))

print()
print("ASIC transistor counts:")
print("  Control Circuits        %5d   (%3d%%)" % (mod_tran["pdep_pext"], round(100*mod_tran["pdep_pext"]/mod_tran["design"])))
print("  Butterfly Networks      %5d   (%3d%%)" % (mod_tran["pdep_pext_butterfly"], round(100*mod_tran["pdep_pext_butterfly"]/mod_tran["design"])))
print("  Parallel Prefix Count   %5d   (%3d%%)" % (mod_tran["pdep_pext_ppc"], round(100*mod_tran["pdep_pext_ppc"]/mod_tran["design"])))
print("  LROTC of Zero Units     %5d   (%3d%%)" % (mod_tran["pdep_pext_decoder"], round(100*mod_tran["pdep_pext_decoder"]/mod_tran["design"])))

print()
print("Size reduction from flattening design hierarchy:")
print("  By gates:       %3d%%" % round((100 - 100*mod_gates["pdep_pext_flat"]/mod_gates["design"])))
print("  By transistors: %3d%%" % round((100 - 100*mod_tran["pdep_pext_flat"]/mod_tran["design"])))

print()
print("Size reduction from eliminating grev operation (flattened):")
print("  By gates:       %3d%%" % round((100 - 100*mod_gates["pdep_pext_nogrev"]/mod_gates["pdep_pext_flat"])))
print("  By transistors: %3d%%" % round((100 - 100*mod_tran["pdep_pext_nogrev"]/mod_tran["pdep_pext_flat"])))

for mod, desc in [["shift64", '"shift64"'], ["rshift64", '"rshift64"'], ["mul32", '"mul32"'], ["pdep_pext_mc3", "multi-cycle pdep_pext"]]:
    print()
    print("Relative size compared to %s:" % desc)
    print("  By gates:       %5.2fx (not flattened), %5.2fx (flattened)" % (mod_gates["design"]/mod_gates[mod], mod_gates["pdep_pext_flat"]/mod_gates[mod]))
    print("  By transistors: %5.2fx (not flattened), %5.2fx (flattened)" % (mod_tran["design"]/mod_tran[mod], mod_tran["pdep_pext_flat"]/mod_tran[mod]))

print()
print("Summary by relative transistor count:")
for mod, desc in [
        ["mul32", '"mul32" reference'],
        ["pdep_pext_flat", "single-cycle pdep_pext"],
        ["shift64", '"shift64" reference'],
        ["mul16", '"mul16" reference'],
        ["pdep_pext_mc3", "multi-cycle pdep_pext"],
        ["grev", 'stand-alone grev'],
        ["rshift64", '"rshift64" reference'],
    ]:
    print("  %-25s %5.2f" % (desc, mod_tran[mod]/mod_tran["pdep_pext_flat"]))

print()
print("Max path from iCE40 timing analysis (no pipelining):")
with open("ice40.rpt") as f:
    for line in f:
        if line.startswith("Total"):
            print("  " + line, end="")

print()
print("Max path from iCE40 timing analysis (with pipelining):")
with open("ice40p.rpt") as f:
    for line in f:
        if line.startswith("Total"):
            print("  " + line, end="")

print()
print("Max path from iCE40 timing analysis (multi-cycle):")
with open("ice40m.rpt") as f:
    for line in f:
        if line.startswith("Total"):
            print("  " + line, end="")

print()

