#!/bin/bash
set -ex
verilator -exe -cc -Wno-fatal test011.v test011.cc 
make -C obj_dir/ -f Vtest011.mk
./obj_dir/Vtest011

iverilog test011_tb.v test011.v
./a.out
