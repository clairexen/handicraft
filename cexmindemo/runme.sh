#!/bin/bash
set -x
rm -f design.{il,aig,aim}
rm -f trace_*.aiw
rm -f trace.vcd

# Process Verilog design and convert to AIG
yosys -Qp '
	read -sv top.sv
	prep -flatten -top top
	hierarchy -simcheck
	write_rtlil design_rtl.il
	check
	opt -full
	techmap
	opt -fast
	dffunmap
	abc -g AND -fast
	opt_clean
	stat
	write_rtlil design_aig.il
	write_aiger -I -B -map design.aim design.aig
'

# Solve with ABC's "bmc3" and write CEX with various write_cex options
# Note: Option -mx will minimize using the 'cexinfo' algorithm.
# Options -m, -mu, -me, and -mo minimize with other algorithms.
cexoptions="c nc ac anc anmc anmxc amc amuc amec amoc amxc nzc nzmxc"
yosys-abc -c 'read_aiger design.aig; fold; strash; bmc3 -F 20 -v' \
    -c "$(for o in $cexoptions; do echo echo\; echo == trace_$o.aiw ==\; write_cex -$o trace_$o.aiw\;; done;) echo"

# Read (non-minimized) CEX and generate VCD by re-simulating CEX on RTL design
yosys -QTp '
	read_rtlil design_rtl.il
	sim -r trace_ac.aiw -map design.aim -clock clock -vcd trace.vcd
'

# Read minimized CEX and generate VCD by re-simulating CEX on AIG design
yosys -QTp '
	read_rtlil design_aig.il
	sim -r trace_amxc.aiw -map design.aim -clock clock -vcd trace_min.vcd
'

# Set initial state based on VCD trace, and print that state
vcd2fst trace.vcd trace.fst
yosys -QTp '
	read_rtlil design_rtl.il
	dump w:init w:state
	sim -r trace.fst -scope top -w
	dump w:init w:state
'

# print different CEX files and the AIGER map file
paste trace_ac.aiw trace_amc.aiw trace_amuc.aiw trace_amec.aiw trace_amoc.aiw trace_amxc.aiw | expand -t20
paste design.aim trace_anmc.aiw trace_anmxc.aiw | expand -t30

# show non-minimized and minimized trace
#twinwave trace.vcd trace.gtkw + trace_min.vcd trace.gtkw

exit 0
