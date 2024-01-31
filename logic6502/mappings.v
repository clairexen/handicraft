
module D_LATCH_001(input D, input E, output Q);
wire A, B, C, F, G, H, I;
assign A = ~D, B = E&A, C = B|H, I = ~C, F = D&E, G = I|F, H = ~G, Q=F|I;
endmodule

module D_LATCH_002(input D, input E, output NOT_Q);
SR sr01 ( .R(D&E), .S(E), .Q(NOT_Q) );
endmodule

