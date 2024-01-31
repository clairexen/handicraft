#!/bin/bash
set -x

yosys -p '
read_verilog -formal counter2.v
prep -top counter
write_smt2 counter2b.smt2'

yosys-smtbmc --dump-smt2 counter2b_compact.smt2 counter2b.smt2
yosys-smtbmc --unroll --dump-smt2 counter2b_unrolled.smt2 counter2b.smt2
