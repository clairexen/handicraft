#!/bin/bash

set -ex

if [ ! -x ./tetgen1.4.3/tetgen ]; then
	tar xvzf tetgen1.4.3.tar.gz
	make -C tetgen1.4.3
fi

./tetgen1.4.3/tetgen cubetest.node
./tetgen1.4.3/tetgen -r -a0.01 -S117 cubetest.1

g++ -std=gnu++0x -o mytetview -lSDL -lGLEW -lm mytetview.cc
# ./mytetview E cubetest.2.node cubetest.2.ele
./mytetview F cubetest.2.node cubetest.2.face

