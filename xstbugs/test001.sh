#!/bin/bash

. /opt/Xilinx/14.5/ISE_DS/settings64.sh

cat > test001.prj <<- EOT
	verilog work "test001_rtl.v"
EOT

cat > test001.xst <<- EOT
	run
	-ifn test001.prj -ofn test001 -p artix7 -top test001
	-iobuf NO -ram_extract NO -rom_extract NO -use_dsp48 NO
EOT

set -ex

xst -ifn test001.xst
netgen -w -ofmt verilog test001.ngc test001_xst.v

vlogcomp test001_xst.v test001_tb.v
fuse -o test001_xst -lib unisims_ver -top test001_tb

vlogcomp test001_rtl.v test001_tb.v
fuse -o test001_rtl -lib unisims_ver -top test001_tb

./test001_xst -tclbatch run-all.txt
./test001_rtl -tclbatch run-all.txt

