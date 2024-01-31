#!/bin/sh
set -ex
g++ -o test.bin test.cc float.cc byte.cc -lm
./test.bin
