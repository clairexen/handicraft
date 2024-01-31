module test_014(a, y);
  input signed [7:0] a;
  output [15:0] y;
  assign y = {3{{~22'd0}}} <<< {4{a}};
endmodule
