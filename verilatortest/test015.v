module test015(a, y);
  input signed [3:0] a;
  output [3:0] y;
  assign y = $signed(5'd1 > a-a);
endmodule
