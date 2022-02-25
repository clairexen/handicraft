#!/bin/bash
set -ex

ccache_log=false
reset_stats=true
build_z3=true

mkdir -p ccachedata usr/bin
export CCACHE_DIR="$PWD/ccachedata"
prefix="$PWD/usr"

if $ccache_log; then
	ccache -o log_file="$PWD/ccachelog.txt"
else
	ccache -o log_file=""
fi

if $reset_stats; then
	ccache -z
fi

export CXX="ccache clang++-12"
export CC="ccache clang-12"

if $build_z3; then
	cd z3
	rm -rf build
	python3 scripts/mk_make.py -d -t -p "$prefix"
	time make -j6 -C build CC="${CC}" CXX="${CXX}"
	make -C build install
	ccache -s
	cd ..
fi

usr/bin/z3 -st -T:3 regression.smt2
