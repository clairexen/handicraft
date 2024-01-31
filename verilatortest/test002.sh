#!/bin/bash
set -ex
verilator -exe -cc -Wno-fatal test002.v test002.cc 
make -C obj_dir/ -f Vtest002.mk
./obj_dir/Vtest002 
