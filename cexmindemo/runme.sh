#!/bin/bash
set -x
rm -f design.{il,aig,aim}
rm -f trace_*.aiw
rm -f trace.vcd

yosys -p '
	read -sv top.sv
	prep -flatten -top top
	hierarchy -simcheck
	check
	opt -full
	techmap
	opt -fast
	dffunmap
	abc -g AND -fast
	opt_clean
	stat
	write_rtlil design.il
	write_aiger -I -B -map design.aim design.aig
'

cexoptions="c nc ac anc anmc anmxc amc amuc amec amoc amxc nzc nzmxc"
yosys-abc -c 'read_aiger design.aig; fold; strash; bmc3 -F 20 -v' \
    -c "$(for o in $cexoptions; do echo echo\; echo == trace_$o.aiw ==\; write_cex -$o trace_$o.aiw\;; done;) echo"

yosys -p '
	read_rtlil design.il
	sim -r trace_amxc.aiw -map design.aim -clock clock -vcd trace.vcd
'

paste trace_ac.aiw trace_amc.aiw trace_amuc.aiw trace_amec.aiw trace_amoc.aiw trace_amxc.aiw | expand -t20
paste design.aim trace_anmc.aiw trace_anmxc.aiw | expand -t30
exit 0
