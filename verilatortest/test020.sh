#!/bin/bash
set -ex
verilator -exe -cc -Wno-fatal test020.v test020.cc 
make -C obj_dir/ -f Vtest020.mk
./obj_dir/Vtest020
iverilog test020_tb.v test020.v
./a.out
