#!/bin/bash
set -ex
eval "$( yosys-config --cxx --cxxflags --ldflags -o nmos_pass.so -shared nmos_pass.cc --ldlibs )"
yosys -l nmos_pass.log -m ./nmos_pass.so nmos_pass.ys
