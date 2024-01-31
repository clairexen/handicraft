#!/bin/bash
. /opt/Xilinx/13.4/ISE_DS/settings64.sh
xst -ifn sincpde_xst.xst
netgen -w -ofmt verilog sincpde.ngc sincpde_synthesis
