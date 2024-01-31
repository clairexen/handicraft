#!/bin/bash
set -ex
verilator -exe -cc -Wno-fatal test014.v test014.cc 
make -C obj_dir/ -f Vtest014.mk
./obj_dir/Vtest014
