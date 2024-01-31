#!/bin/bash
# Usage example: bash disasm.sh 0x00091518 0x00a6459c
set -ex
for w; do
	echo ".word $w"
done > disasm.s
riscv32-unknown-elf-gcc -c disasm.s
riscv32-unknown-elf-objdump -d -M numeric,no-aliases disasm.o
rm disasm.s disasm.o
