#!/bin/bash

set -ex
rm -rf test002
mkdir -p test002
cd test002

cat > test002.v << "EOT"
module test002(y0, y1);
  output [3:0] y0;
  output [3:0] y1;

  // this should return zero (see table 5-5 of IEEE Std 1364-2005)
  assign y0  = -4'd1 ** -4'sd2;
  assign y1  = -4'd1 ** -4'sd3;
endmodule
EOT

cat > test002_tb.v << "EOT"
module tb;
  wire [3:0] y0, y1;
  test002 uut(.y0(y0), .y1(y1));
  initial #100 $display("%b %b", y0, y1);
endmodule
EOT

cat > test002.tcl << "EOT"
read_lib ../cells_cmos.lib
set target_library "../cells_cmos.db"
set synthetic_library "dw_foundation.sldb"
set link_library "* $target_library $synthetic_library"
analyze -format verilog test002.v
elaborate test002
link
uniquify
check_design
compile
write -format verilog -output test002_syn.v
exit
EOT

dc_shell-t -no_gui -f test002.tcl

# use iverilog git b7b77b2 or newer
iverilog -s tb -o test002_pre test002_tb.v test002.v
iverilog -s tb -o test002_post test002_tb.v test002_syn.v ../cells_cmos.v

./test002_pre
./test002_post

