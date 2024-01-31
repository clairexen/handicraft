#!/bin/bash
set -ex
../../dfslang sincpde_multi
UNISIMS=/opt/Xilinx/14.2/ISE_DS/ISE/verilog/src/unisims
iverilog -o sincpde_multi_tb sincpde_multi.v sincpde_multi_tb.v sincpde_div.v sincpde_imac.v ${UNISIMS}/DSP48E1.v ${UNISIMS}/../glbl.v
./sincpde_multi_tb
