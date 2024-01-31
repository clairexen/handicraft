#!/bin/bash

set -ex

# Pass: 2015.4 2016.1 2016.2
# Fail: 2016.3 2016.4 2017.1 2017.2
VIVADO_HLS_VER=2017.2

PATCH_DEMO_V=false
XSIM_START_GUI=false

rm -rf hlsprj vivado_hls.log .Xil work.testbench.wdb xelab.log xelab.pb
rm -rf xsim.dir xsim.jou xsim.log xvlog.log xvlog.pb

/opt/Xilinx/Vivado_HLS/$VIVADO_HLS_VER/bin/vivado_hls -f hlsprj.tcl

if $PATCH_DEMO_V; then
	# Drive .ap_start() on demo_func2 with constant 1. This fixes the issue.
	sed -i "/^demo_func2/,/;/ s/.ap_start(.*)/.ap_start(1'b1)/;" hlsprj/solution/syn/verilog/demo.v
fi

xvlog demo_tb.v hlsprj/solution/syn/verilog/*.v
if $XSIM_START_GUI; then
	xelab --debug all work.testbench
	xsim --gui --view testbench.wcfg work.testbench
else
	xelab -R work.testbench
fi

