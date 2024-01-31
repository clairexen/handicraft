#!/bin/bash
set -ex
verilator -exe -cc -Wno-fatal test018.v test018.cc 
make -C obj_dir/ -f Vtest018.mk
./obj_dir/Vtest018
