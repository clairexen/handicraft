module test (input A, B, C, output X, Y);
	sub sub_inst (
		.I0(A),
		.I1(B),
		.I2(C),
		.XOR_012(X)
	);
	assign Y = sub_inst.XOR_01;
endmodule

module sub (input I0, I1, I2, output XOR_012);
	wire XOR_01 = I0 ^ I1;
	assign XOR_012 = XOR_01 ^ I2;
endmodule
