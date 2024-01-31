module demo03(input [3:0] A, B, C, input S);
	wire [3:0] X = A + (S ? C : B);
	wire [3:0] Y = S ? A+C : A+B;
	assert property (X === Y);
endmodule
