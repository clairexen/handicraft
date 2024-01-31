#!/bin/bash
set -ex
verilator -exe -cc -Wno-fatal test019.v test019.cc 
make -C obj_dir/ -f Vtest019.mk
./obj_dir/Vtest019
