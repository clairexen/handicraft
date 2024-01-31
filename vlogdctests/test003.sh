#!/bin/bash

set -ex
rm -rf test003
mkdir -p test003
cd test003

cat > test003.v << "EOT"
module test003(a, y);
  input [3:0] a;
  output [3:0] y;
  assign y = a >>> 4'bx;
endmodule
EOT

cat > test003.tcl << "EOT"
analyze -format verilog test003.v
elaborate test003
exit
EOT

dc_shell-t -no_gui -f test003.tcl

