#!/bin/bash

. /opt/Xilinx/14.2/ISE_DS/settings64.sh

set -ex
xst -ifn testmodule.xst
ngc2edif -w testmodule.ngc netlist.edif
netgen -w -ofmt verilog testmodule.ngc netlist.v

rm -rf xst testmodule_xst.xrpt testmodule.lso testmodule.srp
rm -rf testmodule.ngc _xmsgs ngc2edif.log testmodule.ngr netlist.nlf

