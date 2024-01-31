#!/bin/bash
set -ex
verilator -exe -cc -Wno-fatal test005.v test005.cc 
make -C obj_dir/ -f Vtest005.mk
# iverilog -o test005_tb test005_tb.v test005.v
# ./test005_tb
./obj_dir/Vtest005 
