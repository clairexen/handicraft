#!/bin/bash

set -ex

# uimode=gui
uimode=console

xvlog testbench.v
xvlog runhls_prj/solution/syn/verilog/*.v
ln -sf runhls_prj/solution/syn/verilog/*.dat .

case "$uimode" in
	gui)
		xelab --debug all work.testbench
		xsim --gui work.testbench
		;;
	console)
		xelab -R work.testbench
		;;
esac

rm -rf .Xil xsim[._]* xelab.* xvlog.* webtalk[._]* *.dat

