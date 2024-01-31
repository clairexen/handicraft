#!/bin/bash
set -ex
../../dfslang sincpde_cores
UNISIMS=/opt/Xilinx/14.2/ISE_DS/ISE/verilog/src/unisims
iverilog -o sincpde_cores_tb sincpde_cores.v sincpde_cores_tb.v sincpde_div.v sincpde_imac.v ${UNISIMS}/DSP48E1.v ${UNISIMS}/../glbl.v
./sincpde_cores_tb
