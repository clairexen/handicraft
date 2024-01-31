#!/bin/bash
set -ex
verilator -exe -cc -Wno-fatal test009.v test009.cc 
make -C obj_dir/ -f Vtest009.mk
./obj_dir/Vtest009

iverilog test009.v
./a.out

/opt/Xilinx/Vivado/2014.1/bin/xvlog test009.v 
/opt/Xilinx/Vivado/2014.1/bin/xelab -R test009

/opt/altera/13.1/modelsim_ase/bin/vlib work
/opt/altera/13.1/modelsim_ase/bin/vlog test009.v 
/opt/altera/13.1/modelsim_ase/bin/vsim -c -do "run; exit" work.test009

yosys -p 'cd test009; eval -show a,b,c,d,e,f,g' test009.v

