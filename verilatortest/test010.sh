#!/bin/bash
set -ex
verilator -exe -cc -Wno-fatal test010.v test010.cc 
make -C obj_dir/ -f Vtest010.mk
./obj_dir/Vtest010
