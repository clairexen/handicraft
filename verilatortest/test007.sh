#!/bin/bash
set -ex
verilator -exe -cc -Wno-fatal test007.v test007.cc 
make -C obj_dir/ -f Vtest007.mk
./obj_dir/Vtest007 
