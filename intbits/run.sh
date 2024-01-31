#!/bin/bash
set -ex
cbmc --trace --function checker s2u.c
cbmc --bounds-check --pointer-overflow-check --nan-check   \
	--div-by-zero-check --conversion-check             \
	--signed-overflow-check --unsigned-overflow-check  \
	--undefined-shift-check --float-overflow-check     \
	--yices --trace --function checker intbits32.c
cbmc --bounds-check --pointer-overflow-check --nan-check   \
	--div-by-zero-check --conversion-check             \
	--signed-overflow-check --unsigned-overflow-check  \
	--undefined-shift-check --float-overflow-check     \
	--yices --trace --function checker intbits64.c
clang -o sim32 -Wall -D SIM intbits32.c
./sim32
clang -o sim64 -Wall -D SIM intbits64.c
./sim64
