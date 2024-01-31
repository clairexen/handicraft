#!/bin/bash
set -ex
MINISAT_DIR="/home/clifford/Temp/minisat/"
clang -MD -std=c++11 -I$MINISAT_DIR -o testsimpsat testsimpsat.cc $MINISAT_DIR/build/release/lib/libminisat.a -lstdc++
grep minisat testsimpsat.d
./testsimpsat
