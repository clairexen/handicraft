module top(input A, B, output Y);
  testbox box(.A(A), .B(B), .Y(Y));
endmodule

module testbox(input A, B, output Y);
endmodule
