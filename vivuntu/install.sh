#!/bin/bash

for dir in /opt/Xilinx/Vivado/$1/bin /opt/Xilinx/Vivado_HLS/$1/bin /opt/Xilinx/SDK/$1/bin /opt/Xilinx/SDK/$1/bin/lin64; do
	[ -d "$dir" ] || continue
	echo; echo "Installing hacks to $dir:"
	for bin in xdg-open firefox kfmclient acroread chromium-browser; do
		rm -f "$dir"/$bin; install -vT wrapper.sh "$dir"/$bin
	done
done

