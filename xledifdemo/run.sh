#!/bin/bash
yosys -p 'synth_xilinx -top top -edif test.edf' test.v
