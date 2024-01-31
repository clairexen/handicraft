module test008(a, b, y);
  input signed [4:0] a;
  input [2:0] b;
  output [3:0] y;
  assign y = |0 != (a >>> b);
endmodule
