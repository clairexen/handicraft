#!/bin/bash

. /opt/Xilinx/13.1/ISE_DS/settings32.sh
set -ex

cd lib_vhdl
xst -ifn "regcell.xst" -ofn "regcell.syr"

cd ../top_verilog
xst -ifn "topmod.xst" -ofn "topmod.syr"

cd ..
ngdbuild -uc example.ucf -p XC9536XL-VQ44 -sd lib_vhdl/ top_verilog/topmod.ngc example.ngd
cpldfit -p XC9536XL-7-VQ44 -ofmt verilog -optimize density -htmlrpt -loc on -slew fast \
        -init low -inputs 32 -pterms 28 -unused float -terminate keeper example.ngd
hprep6 -i example

impact -batch impact.cmd

