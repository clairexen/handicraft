#!/bin/bash
set -ex
verilator -exe -cc -Wno-fatal test015.v test015.cc 
make -C obj_dir/ -f Vtest015.mk
./obj_dir/Vtest015
