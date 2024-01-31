#!/bin/bash

. /opt/Xilinx/12.4/ISE_DS/settings32.sh

set -ex

rm -rf testprog.out
mkdir testprog.out
cd testprog.out

cp ../testprog.v .
cp ../testprog.ucf .

cat > testprog.prj <<- EOT
        verilog work "testprog.v"
EOT

cat > testprog.lso <<- EOT
        work
EOT

cat > testprog.xst <<- EOT
	set -tmpdir "xilinx/projnav.tmp"
	set -xsthdpdir "xilinx"
	run
	-ifn testprog.prj
	-ifmt mixed
	-ofn testprog
	-ofmt NGC
	-p xbr
	-top top
	-opt_mode Speed
	-opt_level 1
	-iuc NO
	-lso testprog.lso
	-keep_hierarchy YES
	-netlist_hierarchy as_optimized
	-rtlview Yes
	-hierarchy_separator /
	-bus_delimiter <>
	-case maintain
	-verilog2001 YES
	-fsm_extract YES -fsm_encoding Auto
	-safe_implementation No
	-mux_extract YES
	-resource_sharing YES
	-iobuf YES
	-pld_mp YES
	-pld_xp YES
	-pld_ce YES
	-wysiwyg NO
	-equivalent_register_removal YES
EOT

cat > testprog.cmd <<- EOT
	setMode -bs
	setCable -port svf -file "testprog.svf"
	addDevice -p 1 -file "testprog.jed"
	# Erase -p 1 
	Program -p 1 -e -v 
	Verify -p 1 
	quit
EOT

cat > erasecpld.cmd <<- EOT
	setMode -bs
	setCable -port svf -file "erasecpld.svf"
	addDevice -p 1 -file "testprog.jed"
	Erase -p 1 
	quit
EOT

mkdir -p xilinx/projnav.tmp/
xst -ifn "testprog.xst" -ofn "testprog.syr"

mkdir -p xilinx/_ngo/
ngdbuild -dd xilinx/_ngo -uc testprog.ucf -p XC9536XL-VQ44 testprog.ngc testprog.ngd

cpldfit -p XC9536XL-7-VQ44 -ofmt verilog -optimize density -htmlrpt -loc on -slew fast \
	-init low -inputs 32 -pterms 28 -unused float -terminate keeper testprog.ngd

hprep6 -i testprog

impact -batch testprog.cmd
sed '\,^// Date:, d;' < testprog.svf > ../testprog.svf

impact -batch erasecpld.cmd
sed '\,^// Date:, d;' < erasecpld.svf > ../erasecpld.svf

