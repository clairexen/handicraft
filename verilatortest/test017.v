module test017(a, y);
  input [4:0] a;
  output [4:0] y;
  assign y = a >> ((a ? 1 : 2) << a);
endmodule
