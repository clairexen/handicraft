#!/bin/bash

set -ex
rm -f runtests.dat

clang -Wall -Wextra -std=c99 -o smtshuffle smtshuffle.c

for i in 1 2 3 4 5 6 7 8; do
	for j in shuffle test1 test2 test3; do
		if [ $j == shuffle ]; then
			{ echo "(set-option :random-seed $i)"; ./smtshuffle $i < test1.smt2; } > temp.smt2
		else
			{ echo "(set-option :random-seed $i)"; cat $j.smt2; } > temp.smt2
		fi
		! /usr/bin/time -o runtests.dat -a --quiet -f "%M %U seed$i $j" z3 temp.smt2
		rm -f temp.smt2
	done
done

