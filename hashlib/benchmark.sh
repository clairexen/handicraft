#!/bin/bash

cfg_n="100 1000 10000 100000 1000000 10000000"
cfg_m="20000000"

set -ex
clang -o benchmark-clang -std=c++11 -Wall -Os benchmark.cc -lstdc++ -lm
gcc -o benchmark-gcc -std=c++11 -Wall -Os benchmark.cc -lstdc++ -lm
exec > benchmark.dat

for impl in set map unordered_set unordered_map dict pool none; do
for type in int string; do for mode in dense sparse; do for comp in clang gcc; do for N in $cfg_n; do
	echo $comp $impl $type $mode $N $(/usr/bin/time -f '%U %M' ./benchmark-$comp $impl $type $mode $N $cfg_m 2>&1)
done; done; done; done; done

python benchmark.py

