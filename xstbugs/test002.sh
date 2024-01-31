#!/bin/bash

. /opt/Xilinx/14.5/ISE_DS/settings64.sh

cat > test002.prj <<- EOT
	verilog work "test002_rtl.v"
EOT

cat > test002.xst <<- EOT
	run
	-ifn test002.prj -ofn test002 -p artix7 -top test002
	-iobuf NO -ram_extract NO -rom_extract NO -use_dsp48 NO
EOT

set -ex

xst -ifn test002.xst
netgen -w -ofmt verilog test002.ngc test002_xst.v

vlogcomp test002_xst.v test002_tb.v
fuse -o test002_xst -lib unisims_ver -top test002_tb

vlogcomp test002_rtl.v test002_tb.v
fuse -o test002_rtl -lib unisims_ver -top test002_tb

./test002_xst -tclbatch run-all.txt
./test002_rtl -tclbatch run-all.txt

