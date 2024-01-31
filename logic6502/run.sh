#!/bin/bash

# run switch-level simutation
clang -o  switchsim -std=c++11 switchsim.cc -lstdc++ -O0 -ggdb
./switchsim

# convert switches to set-reset-netlist
bash sw2sr.sh

# generate and run set-reset simutation
bash simgen.sh
./sim

