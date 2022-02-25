#!/bin/bash
set -ex
mkdir -p ccachedata
export CCACHE_DIR="$PWD/ccachedata"
export CXX="ccache clang++-12"
export CC="ccache clang-12"
ccache -s
