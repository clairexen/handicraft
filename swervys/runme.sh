#!/bin/bash
if ! test -d swerv_eh1; then
	git clone https://github.com/westerndigitalcorporation/swerv_eh1
	( cd swerv_eh1; patch -p1 < ../swerv.diff; )
fi
yosys -ql swerv.log swerv.ys
