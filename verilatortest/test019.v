module test019(a, y);
  input [3:0] a;
  output [3:0] y;
  assign y = (a >> a) ^~ (a >> a);
endmodule
