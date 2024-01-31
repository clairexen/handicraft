#!/bin/bash

vivado -mode batch -source vivadobug03.tcl

echo
echo "==================== Pre-Synthesis Simulation ===================="
echo

xvlog vivadobug03_tb.v
xvlog vivadobug03.v 
xelab -R testbench

echo
echo "==================== Post-Synthesis Simulation ===================="
echo

xvlog glbl.v
xvlog vivadobug03_tb.v
xvlog vivadobug03_syn.v 
xelab -L unisims_ver -R testbench glbl

