#!/bin/bash
set -x
sby -f abc_bmc3.sby
cexoptions="ac anc anmc amc amuc amec amoc amxc"
yosys-abc -c 'read_aiger abc_bmc3/model/design_aiger.aig; fold; strash; bmc3 -F 20 -v' \
    -c "$(for o in $cexoptions; do echo write_cex -$o trace_$o.aiw\;; done;)"
cat trace_ac.aiw
cat trace_amc.aiw
cat trace_amuc.aiw
cat trace_amxc.aiw
exit 0
