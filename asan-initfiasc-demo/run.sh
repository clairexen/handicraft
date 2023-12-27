#!/bin/bash
set -ev
CXX="clang++"
CXXFLAGS="-g -fsanitize=address"
#CXXFLAGS="${CXXFLAGS} -std=c++17"     # ./pass fails with this
CXXFLAGS="${CXXFLAGS} -std=c++20"      # ./pass fails without this
CXXFLAGS="${CXXFLAGS} -stdlib=libc++"  # ./pass fails without this
${CXX} ${CXXFLAGS} -D ASAN_INIT_ORDER_EXPECT_PASS -g -o pass part1.cc part2.cc
${CXX} ${CXXFLAGS} -D ASAN_INIT_ORDER_EXPECT_FAIL -g -o fail part1.cc part2.cc
ASAN_OPTIONS=check_initialization_order=true ./pass || { echo 'Failed ./pass !!!'; exit 1; }
ASAN_OPTIONS=check_initialization_order=true ./fail && { echo 'Passed ./fail !!!'; exit 1; }
echo OKAY
