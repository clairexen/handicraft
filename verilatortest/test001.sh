#!/bin/bash
set -ex
verilator -exe -cc -Wno-fatal test001.v test001.cc 
make -C obj_dir/ -f Vtest001.mk
./obj_dir/Vtest001 
