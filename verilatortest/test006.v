module test006(a, y);
  input [3:0] a;
  output [5:0] y;

  localparam signed [3:0] p1 = 4'b1000;
  localparam signed [3:0] p2 = 0;
  assign y = a + (p1 + p2);
endmodule
