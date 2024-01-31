#!/bin/bash
set -ex
../../dfslang sincpde_single
UNISIMS=/opt/Xilinx/14.2/ISE_DS/ISE/verilog/src/unisims
iverilog -o sincpde_single_tb sincpde_single.v sincpde_single_tb.v sincpde_div.v sincpde_imac.v ${UNISIMS}/DSP48E1.v ${UNISIMS}/../glbl.v
./sincpde_single_tb
