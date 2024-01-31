#!/bin/bash
set -ex
MINISAT_DIR="/home/clifford/Temp/minisat/"
MINISAT_LDLIBS="$MINISAT_DIR/build/debug/lib/libminisat.a"
clang -MD -ggdb -std=c++11 -I$MINISAT_DIR -DSKIP_FIRST -o testcase_ok testcase.cc $MINISAT_LDLIBS -lstdc++
clang -MD -ggdb -std=c++11 -I$MINISAT_DIR -o testcase testcase.cc $MINISAT_LDLIBS -lstdc++
./testcase_ok
./testcase
