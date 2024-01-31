#!/bin/bash
yosys -p 'read_verilog -formal counter2.v; prep; write_smt2 -tpl counter2a.tpl counter2a.smt2'
z3 -smt2 counter2a.smt2 
