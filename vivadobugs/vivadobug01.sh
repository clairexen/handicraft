#!/bin/bash

vivado -mode batch -source vivadobug01.tcl

echo
echo "==================== Pre-Synthesis Simulation ===================="
echo

xvlog vivadobug01_tb.v
xvlog vivadobug01.v 
xelab -R testbench

echo
echo "==================== Post-Synthesis Simulation ===================="
echo

xvlog glbl.v
xvlog vivadobug01_tb.v
xvlog vivadobug01_syn.v 
xelab -L unisims_ver -R testbench glbl

