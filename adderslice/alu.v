module alu_gold (
	input m,
	input [7:0] a, b,
	output [8:0] s
);
	assign s = m ? a-b : a+b;
endmodule

module alu_gate (
	input m,
	input [7:0] a, b,
	output [8:0] s
);
	wire [7:0] co;

	alu_slice slices [7:0] (
		.A(a),
		.B(b),
		.CI({co[6:0], m}),
		.M(m),
		.S(s[7:0]),
		.CO(co)
	);

	assign s[8] = co[7] ^ m;
endmodule
