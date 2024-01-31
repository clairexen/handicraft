#!/bin/bash

set -ex
make firmware.hex

mode=1
case $mode in
	1)
		iverilog -o testbench.exe -s testbench testbench.v \
				c3demo.v ledpanel2.v picorv32.v \
				$(yosys-config --datdir/ice40/cells_sim.v)
		;;
	2)
		yosys -l synth.log -p 'synth_ice40 -top c3demo' -o synth.v \
				c3demo.v ledpanel2.v picorv32.v
		iverilog -o testbench.exe -s testbench testbench.v synth.v \
				$(yosys-config --datdir/ice40/cells_sim.v --datdir/simlib.v)
		;;
esac

chmod -x testbench.exe
vvp -N testbench.exe
