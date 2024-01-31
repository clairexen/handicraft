module test013(y);
  output [3:0] y;
  localparam [3:0] p11 = 1'bx;
  assign y = ~&p11;
`ifndef VERILATOR
 initial #1 $display("%1d", y);
`endif
endmodule
