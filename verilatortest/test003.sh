#!/bin/bash
set -ex
verilator -exe -cc -Wno-fatal test003.v test003.cc 
make -C obj_dir/ -f Vtest003.mk
./obj_dir/Vtest003 
