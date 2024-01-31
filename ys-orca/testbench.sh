#!/bin/bash
set -ex
iverilog -o testbench testbench.v synth_ice40.v /usr/local/share/yosys/ice40/cells_sim.v
vvp -N testbench
