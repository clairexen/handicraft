#!/bin/bash
set -ex
covered_vpi=/usr/lib/x86_64-linux-gnu/covered/covered.vpi
rm -f cov.cdd covered_vpi.tab covered_vpi.v db.cdd
covered score -ep -t serial_alu_tb -v serial_alu_tb.v -v serial_alu.v -o db.cdd -vpi
iverilog -o serial_alu_tb serial_alu_tb.v serial_alu.v covered_vpi.v -m $covered_vpi
vvp -N ./serial_alu_tb
covered report -d v cov.cdd
