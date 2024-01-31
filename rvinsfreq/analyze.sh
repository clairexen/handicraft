#!/bin/bash

set -ex

while read fn; do
	riscv64-unknown-elf-objdump -M numeric,no-aliases -d $fn | egrep '^ +[0-9a-f]+:\s+[0-9a-f]+' | awk '{ print $3,$4; }' | \
			sed -r 's/([,( ])([fx][0-9]{1,2}|-?[0-9]+|(0x)?[0-9a-f]+)/\1?/g;'
done < <(find -type f '(' -perm /111 -o -name '*.so' -o -name '*.so.*' ')' | xargs file | grep ELF | cut -f1 -d:) | sort | uniq -c | sort -n
