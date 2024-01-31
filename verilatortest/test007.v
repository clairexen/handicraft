module test007(a, y);
  input signed [3:0] a;
  output [4:0] y;
  assign y = a << -2'sd1;
endmodule
