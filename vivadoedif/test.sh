#!/bin/bash
vivado -mode batch -source test.tcl
yosys -p 'synth_xilinx -edif test_yosys.edn' test.v
