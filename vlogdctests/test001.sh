#!/bin/bash

set -ex
rm -rf test001
mkdir -p test001
cd test001

cat > test001.v << "EOT"
module test001(a, b, c, d, y1, y2, y3);
  input [3:0] a;
  input [3:0] b;
  input [3:0] c;
  input [3:0] d;
  output [3:0] y1;
  output [3:0] y2;
  output [3:0] y3;

  // This should return y=15 for a=15, b=15, c=15, d=15.
  // But Design Compiler G-2012.06-SP4 returns y=7 instead.
  assign y1 = a >>> ((b == c) >>> d);

  // This should return 1 but DC G-2012.06-SP4 returns 0 instead.
  assign y2 = (|1 >>> 1) == |0;

  // This should return 0 but DC G-2012.06-SP4 returns 1 instead.
  assign y3 = {|1 >>> 1};
endmodule
EOT

cat > test001_tb.v << "EOT"
module tb;
  reg [3:0] a = 15, b = 15, c = 15, d = 15;
  wire [3:0] y1, y2, y3;
  test001 uut(.a(a), .b(b), .c(c), .d(d), .y1(y1), .y2(y2), .y3(y3));
  initial #100 $display("%b %b %b", y1, y2, y3);
endmodule
EOT

cat > test001.tcl << "EOT"
read_lib ../cells_cmos.lib
set target_library "../cells_cmos.db"
set synthetic_library "dw_foundation.sldb"
set link_library "* $target_library $synthetic_library"
analyze -format verilog test001.v
elaborate test001
link
uniquify
check_design
compile
write -format verilog -output test001_syn.v
exit
EOT

dc_shell-t -no_gui -f test001.tcl

# use iverilog git b7b77b2 or newer
iverilog -s tb -o test001_pre test001_tb.v test001.v
iverilog -s tb -o test001_post test001_tb.v test001_syn.v ../cells_cmos.v

./test001_pre
./test001_post

