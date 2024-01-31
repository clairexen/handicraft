#!/bin/bash
set -ex
verilator -exe -cc -Wno-fatal test016.v test016.cc 
make -C obj_dir/ -f Vtest016.mk
./obj_dir/Vtest016
