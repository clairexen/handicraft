#!/bin/bash

set -ex

if ! test -d debian-riscv64-tarball-20180418; then
	rm -f debian-riscv64-tarball-20180418.tar.gz
	wget https://people.debian.org/~mafm/debian-riscv64-tarball-20180418.tar.gz
	! tar xvzf debian-riscv64-tarball-20180418.tar.gz
fi

mkdir -p database
cd debian-riscv64-tarball-20180418

while read fn; do
	id=$(echo $fn | md5sum | cut -f1 -d' ')
	{ echo $fn; riscv64-unknown-elf-objdump -M numeric,no-aliases -d $fn; } > ../database/$id.txt
done < <(find -type f '(' -perm /111 -o -name '*.so' -o -name '*.so.*' ')' | xargs file | grep ELF | cut -f1 -d:) | sort | uniq -c | sort -n

