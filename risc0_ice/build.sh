#!/bin/bash
set -ex
yosys -v2 -l output.log RISC0Top.ys
arachne-pnr -d 8k -o output.txt output.blif
icepack output.txt output.bin
