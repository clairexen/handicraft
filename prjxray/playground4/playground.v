module top(input A, B, C, D, E, F, output Y);
  LUT6 #(.INIT(64'h8000000000000000)) mylut (.O(Y), .I0(A), .I1(B), .I2(C), .I3(D), .I4(E), .I5(F));
endmodule
