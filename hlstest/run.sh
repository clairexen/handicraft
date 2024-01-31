#!/bin/bash

set -ex
rm -rf .Xil/
rm -rf mycore.v
rm -rf mycore_hlsprj/
rm -rf vivado_hls.log
rm -rf synth.log

vivado_hls -f mycore.tcl
cat mycore_hlsprj/my_solution/syn/verilog/*.v > mycore.v
vivado -nojournal -log synth.log -mode batch -source synth.tcl

set +x
echo

echo "Vivado HLS Timing Estimate"
echo "=========================="
echo
grep -A6 '^+ Timing' mycore_hlsprj/my_solution/syn/report/mycore_csynth.rpt
echo

echo "Vivado Synthesis Timing"
echo "======================="
echo
grep -A9 ^Slack synth.log
echo

