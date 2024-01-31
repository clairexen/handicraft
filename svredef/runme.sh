#!/bin/bash

set -x

yosys -p "
  verific -sv top.sv
  verific -sv module.sv
  verific -import top
"

yosys -p "
  verific -sv module.sv top.sv
  verific -import top
"

yosys -p "
  verific -sv module.sv top.sv
  verific -import -all
"

cat > vivado.tcl << EOT
read_verilog top.sv
read_verilog module.sv
synth_design -rtl -top top
EOT

PATH=/opt/Xilinx/Vivado/2017.2/bin:$PATH

vivado -mode batch -source vivado.tcl

PATH=/opt/intelFPGA_lite/17.0/modelsim_ase/linux:$PATH

vlib work_mfcu
vlog -work work_mfcu -O1 -mfcu -sv top.sv module.sv

vlib work_sfcu
vlog -work work_sfcu -O1 -sfcu -sv top.sv module.sv
vsim -work work_sfcu -c -do "run -all; exit" work_sfcu.top

