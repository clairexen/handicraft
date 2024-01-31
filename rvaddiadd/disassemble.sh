#!/bin/bash

set -ex

while read fn; do
	riscv64-unknown-elf-objdump -M numeric,no-aliases -d $fn
done < <(find -type f '(' -perm /111 -o -name '*.so' -o -name '*.so.*' ')' | xargs file | grep ELF | cut -f1 -d:)
