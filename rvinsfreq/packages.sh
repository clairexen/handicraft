#!/bin/bash
set -ex

if ! test -f Packages.gz; then
	wget -O Packages.gz http://ftp.ports.debian.org/debian-ports/dists/sid/main/binary-riscv64/Packages.gz
fi

declare -A pindex
while read pkg fn; do
	test ${pkg%-dbgsym} == $pkg || continue
	test -n "${pindex[$pkg]}" && continue
	echo "all:: database/${pkg}.txt"
	echo "database/${pkg}.txt:"
	echo "	bash package.sh $pkg $fn"
	pindex[$pkg]=1
done < <( zcat Packages.gz | gawk '$1 == "Package:" { p = $2; } $1 == "Architecture:" { a = $2; } $1 == "Filename:" && a == "riscv64" { print p, $2; }' | sort -u ) > Makefile
