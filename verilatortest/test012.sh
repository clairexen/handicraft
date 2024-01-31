#!/bin/bash
set -ex
verilator -exe -cc -Wno-fatal test012.v test012.cc 
make -C obj_dir/ -f Vtest012.mk
./obj_dir/Vtest012
