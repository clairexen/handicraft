#!/bin/bash

vivado -mode batch -source vivadobug02.tcl

echo
echo "==================== Pre-Synthesis Simulation ===================="
echo

xvlog vivadobug02_tb.v
xvlog vivadobug02.v 
xelab -R testbench

echo
echo "==================== Post-Synthesis Simulation ===================="
echo

xvlog glbl.v
xvlog vivadobug02_tb.v
xvlog vivadobug02_syn.v 
xelab -L unisims_ver -R testbench glbl

