#!/bin/bash

set -ex

# set seed=1 for bug and seed=2 for no bug
seed=1

yosys -ql syn_out.log -p 'synth_ice40 -blif syn_out.blif; write_verilog syn_out.v' top.v freq_div.v
arachne-pnr -s $seed -d 1k -o pnr_out.txt -p icestick.pcf syn_out.blif
icepack pnr_out.txt pnr_out.bin
iceprog -S pnr_out.bin
exit

iverilog -o sim_rtl.exe -D VCD_FILE='"sim_rtl.vcd"' -s top_tb top.v freq_div.v top_tb.v
vvp -N sim_rtl.exe # -lxt2

iverilog -o sim_syn.exe -D VCD_FILE='"sim_syn.vcd"' -s top_tb syn_out.v top_tb.v $(yosys-config --datdir/ice40/cells_sim.v)
vvp -N sim_syn.exe # -lxt2

icebox_vlog -cn top -p icestick.pcf pnr_out.txt > pnr_out.v
iverilog -o sim_pnr.exe -D VCD_FILE='"sim_pnr.vcd"' -s top_tb pnr_out.v top_tb.v
vvp -N sim_pnr.exe # -lxt2

