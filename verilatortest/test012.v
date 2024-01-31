module test012(a, y);
  input [2:0] a;
  output [3:0] y;
  assign y = (a >> 2'b11) >> 1;
endmodule
