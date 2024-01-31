module test005(a, y);
  input [3:0] a;
  output [3:0] y;
  assign y = ~|a;
endmodule
