#!/bin/bash
set -ex

enable_log=false
reset_stats=true

mkdir -p ccachedata usr/bin
export CCACHE_DIR="$PWD/ccachedata"
prefix="$PWD/usr"

if $enable_log; then
	ccache -o log_file="$PWD/ccachelog.txt"
else
	ccache -o log_file=""
fi

if $reset_stats; then
	ccache -z
fi

export CXX="ccache clang++-12"
export CC="ccache clang-12"

cd z3
rm -rf build
python3 scripts/mk_make.py -d -t -p "$prefix"
time make -j6 -C build CC="${CC}" CXX="${CXX}"
ccache -s
