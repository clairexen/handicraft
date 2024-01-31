#!/bin/bash
iverilog -o testbench testbench.v reference.v
./testbench
