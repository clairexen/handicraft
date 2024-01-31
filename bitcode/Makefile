#!/bin/bash

notneg.txt: notneg.py aggregate.dump perm_b32.dump swartmps.dump bithacks.dump bzip2.dump oggenc.dump
	python3 notneg.py > notneg.txt

aggregate.o: aggregate.c
	riscv32-unknown-elf-gcc -march=rv32g -c -Wall -Os aggregate.c

perm_b32.o: perm_b32.c
	riscv32-unknown-elf-gcc -march=rv32g -c -Wall -Os perm_b32.c

swartmps.o: swartmps.cpp
	riscv32-unknown-elf-gcc -march=rv32g -c -Wall -Wno-sign-compare -Os swartmps.cpp

bithacks.o: bithacks.c
	riscv32-unknown-elf-gcc -march=rv32g -c -Wall -Os bithacks.c

bzip2.o: bzip2.c
	riscv32-unknown-elf-gcc -march=rv32g -c -Wall -Wno-unused-but-set-variable -Wno-implicit-function-declaration -Os bzip2.c

oggenc.o: oggenc.c
	riscv32-unknown-elf-gcc -march=rv32g -c -Wall -Wno-address -Wno-unused-but-set-variable -Wno-unused-variable -Wno-maybe-uninitialized -Os oggenc.c

%.dump: %.o
	riscv32-unknown-elf-objdump -d $< > $@

