#!/bin/bash
set -ev
CXX="clang++"
CXXFLAGS="-g -fsanitize=address"
#CXXFLAGS="${CXXFLAGS} -std=c++17"     # ./pass fails with this
CXXFLAGS="${CXXFLAGS} -std=c++20"      # ./pass fails without this
CXXFLAGS="${CXXFLAGS} -stdlib=libc++"  # ./pass fails without this
${CXX} ${CXXFLAGS} -D ASAN_INIT_ORDER_EXPECT_FAIL -g -S -o fail_part2.s part2.cc
${CXX} ${CXXFLAGS} -D ASAN_INIT_ORDER_EXPECT_PASS -g -S -o pass_part2.s part2.cc
${CXX} ${CXXFLAGS} -D ASAN_INIT_ORDER_CONSTINIT -g -S -o constinit_part2.s part2.cc
${CXX} ${CXXFLAGS} -D ASAN_INIT_ORDER_EXPECT_FAIL -g -o fail part1.cc part2.cc
${CXX} ${CXXFLAGS} -D ASAN_INIT_ORDER_EXPECT_PASS -g -o pass part1.cc part2.cc
${CXX} ${CXXFLAGS} -D ASAN_INIT_ORDER_CONSTINIT -g -o constinit part1.cc part2.cc
ASAN_OPTIONS=print_legend=false:check_initialization_order=true ./fail && { echo 'Passed ./fail !!!'; exit 1; }
ASAN_OPTIONS=check_initialization_order=true ./pass || { echo 'Failed check_initialization_order ./pass !!!'; exit 1; }
ASAN_OPTIONS=check_initialization_order=true ./constinit || { echo 'Failed check_initialization_order ./constinit !!!'; exit 1; }
ASAN_OPTIONS=strict_init_order=true ./pass || { echo 'Failed strict_init_order ./pass !!!'; exit 1; }
ASAN_OPTIONS=strict_init_order=true ./constinit || { echo 'Failed strict_init_order ./constinit !!!'; exit 1; }
! diff -u pass_part2.s constinit_part2.s
echo OKAY
