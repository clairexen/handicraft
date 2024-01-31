#!/bin/bash
( set -ex
../../dfslang det3
iverilog det3.v det3_tb.v
./a.out | tee output.txt; )
gawk '	i == 0 && /^det_1: *-20$/ { i++; next; }
	i == 1 && /^det_2: *20$/ { i++; next; }
	i == 2 && /^det_M: *-400$/ { i++; next; }
	i == 3 && /^det_1: *0$/ { i++; next; }
	i == 4 && /^det_2: *0$/ { i++; next; }
	i == 5 && /^det_M: *0$/ { i++; next; }
	END { exit(i != 6); }' output.txt
