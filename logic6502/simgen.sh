#!/bin/bash
set -ex
eval "$( yosys-config --cxx --cxxflags --ldflags -o simgen.so -shared simgen.cc --ldlibs )"
yosys -l simgen.log -m ./simgen.so -p opt -p 'simgen MOS6502' logic6502.v
clang -o sim -Wall -Wextra -Werror -O3 sim_main.cc sim_MOS6502.cc -lstdc++
