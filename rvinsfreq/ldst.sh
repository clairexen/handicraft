#!/bin/bash
for x in debian-riscv64-tarball-20180418/{,usr/}{bin,sbin}/*; do
	riscv64-unknown-elf-objdump -d $x | egrep '^\s+[0-9a-f]+:\s+[0-9a-f]+\s+[ls][bhwd]u?\s'
done | awk '{ print $3, $4; }' > ldst.txt
