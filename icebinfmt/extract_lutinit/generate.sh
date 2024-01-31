#!/bin/bash

mkdir -p binfiles
mkdir -p tmpfiles

{
	echo "all: all_targets"
	make_targets=""

	for i in {0..99}; do
		ii=$(printf "%02d" $i)
		echo "binfiles/ex_${ii}.bin: generate.sh gensingle.sh"
		echo "	bash gensingle.sh $ii"
		echo "	icecuberun tmpfiles/ex_${ii} &> tmpfiles/ex_${ii}.log"
		echo "	{ echo 'obase=16; ibase=2;'; grep LUT_INIT tmpfiles/ex_${ii}.tmp/outputs/simulation_netlist/top_sbt.v | sed -r 's/.*defparam alice_x(..)_y(..)_z(.).*=16.b(.*);.*/print" \
				"\"clb_\\1_\\2.ble_\\3.lutinit 16 #\"; \4 + 10000000000000000/'; } | bc | sed 'y/ABCDEF/abcdef/; s/#1/0x/;' > binfiles/ex_${ii}.val"
		echo "	mv tmpfiles/ex_${ii}.bin binfiles/"
		make_targets="$make_targets binfiles/ex_${ii}.bin"
	done

	echo "all_targets:$make_targets"
} > tmpfiles/makefile

make -j4 -f tmpfiles/makefile

