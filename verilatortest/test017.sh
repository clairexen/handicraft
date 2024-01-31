#!/bin/bash
set -ex
verilator -exe -cc -Wno-fatal test017.v test017.cc 
make -C obj_dir/ -f Vtest017.mk
./obj_dir/Vtest017
