module test (
	input [31:0] A, B,
	input [4:0] shamt
);
	wire [31:0] H, L, R, Y;
	wire reverse;

	`UUT uut (
		.H(H),
		.L(L),
		.shamt(shamt),
		.reverse(reverse),
		.Y(Y)
	);

`ifdef SRL
	assign H = 0, L = A, reverse = 0, R = A >> shamt;
`endif
`ifdef SRA
	assign H = {32{A[31]}}, L = A, reverse = 0, R = $signed(A) >>> shamt;
`endif
`ifdef SLL
	assign H = 0, L = A, reverse = 1, R = A << shamt;
`endif
`ifdef SRO
	assign H = ~0, L = A, reverse = 0, R = ~(~A >> shamt);
`endif
`ifdef SLO
	assign H = ~0, L = A, reverse = 1, R = ~(~A << shamt);
`endif
`ifdef ROR
	assign H = A, L = A, reverse = 0, R = {A,A} >> shamt;
`endif
`ifdef ROL
	assign H = A, L = A, reverse = 1, R = ({A,A} << shamt) >> 32;
`endif
`ifdef FSR
	assign H = B, L = A, reverse = 0, R = {B, A} >> shamt;
`endif
`ifdef FSL
	assign H = B, L = A, reverse = 1, R = ({A, B} << shamt) >> 32;
`endif

	always @* assert(R == Y);
endmodule
