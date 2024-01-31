#!/bin/bash

vivado_hls -f hlsbugtst5.tcl
cp -v hlsbugtst5/solution/syn/verilog/* .

iverilog -o hlsbugtst5.vvp hlsbugtst5_tb.v hlsbugtst5.v hlsbugtst5_prime_table_V.v
./hlsbugtst5.vvp

# /opt/altera/15.1/modelsim_ase/bin/vlib work
# /opt/altera/15.1/modelsim_ase/bin/vlog -vlog01compat hlsbugtst5_tb.v 
# /opt/altera/15.1/modelsim_ase/bin/vlog -vlog01compat hlsbugtst5.v 
# /opt/altera/15.1/modelsim_ase/bin/vlog -vlog01compat hlsbugtst5_prime_table_V.v 
# /opt/altera/15.1/modelsim_ase/bin/vsim -c -do "run -all; exit" work.testbench

