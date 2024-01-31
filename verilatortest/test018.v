module test018(a, b, y);
  input [3:0] a;
  input [5:0] b;
  output [3:0] y;
  assign y = 64'd0 | (a << b);
endmodule
