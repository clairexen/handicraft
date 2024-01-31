#!/bin/bash
set -ex
verilator -exe -cc -Wno-fatal test013.v test013.cc 
make -C obj_dir/ -f Vtest013.mk
./obj_dir/Vtest013
iverilog test013.v
./a.out
