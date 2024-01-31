#!/bin/bash

set -ex

if ! test -d debian-riscv64-tarball-20180418; then
	rm -f debian-riscv64-tarball-20180418.tar.gz
	wget https://people.debian.org/~mafm/debian-riscv64-tarball-20180418.tar.gz
	! tar xvzf debian-riscv64-tarball-20180418.tar.gz
fi

mkdir -p database
cd debian-riscv64-tarball-20180418
bash ../disassemble.sh > ../database/debian-base.txt
