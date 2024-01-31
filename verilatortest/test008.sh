#!/bin/bash
set -ex
verilator -exe -cc -Wno-fatal test008.v test008.cc 
make -C obj_dir/ -f Vtest008.mk
./obj_dir/Vtest008 
