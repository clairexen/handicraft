#!/bin/bash
set -ex
verilator -exe -cc -Wno-fatal test006.v test006.cc 
make -C obj_dir/ -f Vtest006.mk
./obj_dir/Vtest006 
