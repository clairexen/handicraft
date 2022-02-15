#!/bin/bash
set -x
sby -f abc_bmc3.sby
cexoptions="c nc ac anc anmc amc amuc amec amoc amxc"
yosys-abc -c 'read_aiger abc_bmc3/model/design_aiger.aig; fold; strash; bmc3 -F 20 -v' \
    -c "$(for o in $cexoptions; do echo write_cex -$o trace_$o.aiw\;; done;)"
paste trace_ac.aiw trace_amc.aiw trace_amuc.aiw trace_amec.aiw trace_amoc.aiw trace_amxc.aiw | expand -t20
exit 0
