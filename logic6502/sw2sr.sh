#!/bin/bash
set -ex
eval "$( yosys-config --cxx --cxxflags --ldflags -o sw2sr.so -shared sw2sr.cc --ldlibs )"
yosys -o mappings.il mappings.v -p opt
yosys -l sw2sr.log -m ./sw2sr.so sw2sr.ys
