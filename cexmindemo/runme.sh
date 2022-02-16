#!/bin/bash
set -x
rm -f trace_*.aiw
#sby -f abc_bmc3.sby

yosys -p '
read -sv top.sv
prep -flatten -top top
hierarchy -simcheck
check
setattr -unset keep
delete -output
opt -full
techmap
opt -fast
dffunmap
abc -g AND -fast
opt_clean
stat
write_aiger -I -B -zinit -map aiger.aim aiger.aig
'

cexoptions="c nc ac anc anmc anmxc amc amuc amec amoc amxc nzc nzmxc"
yosys-abc -c 'read_aiger aiger.aig; fold; strash; bmc3 -F 20 -v' \
    -c "$(for o in $cexoptions; do echo write_cex -$o trace_$o.aiw\;; done;)"

paste trace_ac.aiw trace_amc.aiw trace_amuc.aiw trace_amec.aiw trace_amoc.aiw trace_amxc.aiw | expand -t20
paste trace_anmc.aiw trace_anmxc.aiw | expand -t30
exit 0
