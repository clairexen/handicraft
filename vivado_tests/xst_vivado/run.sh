#!/bin/bash

set -ex
PATH=/opt/Xilinx/14.2/ISE_DS/ISE/bin/lin64:$PATH
PATH=/opt/Xilinx/Vivado/2014.1/bin:$PATH

xst -ifn wxzip_xst.xst
ngc2edif -w wxzip.ngc
vivado -mode batch -source vivado.tcl
