#!/bin/bash

[ "$( uname -m )" = x86_64 ] && xst_bits="64" || xst_bits="32"
xst_version=13.4

make -C ../..
. /opt/Xilinx/$xst_version/ISE_DS/settings$xst_bits.sh

set -xe

xst -ifn example.xst
netgen -w -ofmt verilog example.ngc example_netlist

sed '/module example/,/endmodule/ p; d;' < example_netlist.v | ../../vlnlp-demo > example_vlnlp.v
sed '/module example/,/endmodule/ d;' < example_netlist.v > example_glbl.v

vlogcomp example_tb.v
vlogcomp example_vlnlp.v
vlogcomp example_glbl.v
fuse -o example_tb -lib unisims_ver -top example_tb -top glbl
./example_tb -tclbatch run-all.txt

