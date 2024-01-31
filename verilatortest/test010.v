module test010(y);
  output [4:0] y;
  assign y = { -4'sd7 };
`ifndef VERILATOR
 initial #1 $display("%b", y);
`endif
endmodule
