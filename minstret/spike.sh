#!/bin/bash
LD_LIBRARY_PATH="./riscv-isa-sim:./riscv-fesvr" ./riscv-isa-sim/spike "$@"
