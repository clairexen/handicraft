#!/bin/bash
set -ex
verilator -exe -cc -Wno-fatal test004.v test004.cc 
make -C obj_dir/ -f Vtest004.mk
# iverilog -o test004_tb test004_tb.v test004.v
# ./test004_tb
./obj_dir/Vtest004 
