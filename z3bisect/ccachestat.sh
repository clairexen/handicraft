#!/bin/bash
( set -ex

mkdir -p ccachedata usr/bin
export CCACHE_DIR="$PWD/ccachedata"
prefix="$PWD/usr"

export CXX="ccache clang++-12"
export CC="ccache clang-12"

export FORCE_PS1="\e[0;35mCCACHE:Z3\e[m $PS1"
ccache -s
"$@"; )
