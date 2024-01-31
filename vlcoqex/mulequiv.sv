// The property below passes if N <= M and fails if N > M.
`define N 4
`define M 4

module top (
	input [`N+`M-1:0] A, B,
	output [`N+`M-1:0] D
);
	wire [`N+`M-1:0] X, Y;

	mul1 mul1_inst (A, B, X);
	mul2 mul2_inst (A, B, Y);

	assign D = X - Y;
endmodule

module mul1 (
	input [`N+`M-1:0] A, B,
	output [`N+`M-1:0] P
);
	assign P = A * B;
endmodule

module mul2 (
	input [`N+`M-1:0] A, B,
	output [`N+`M-1:0] P
);
	wire [`N-1:0] a_hi = A >> `M;
	wire [`N-1:0] b_hi = B >> `M;

	wire [`M-1:0] a_lo = A;
	wire [`M-1:0] b_lo = B;

	assign P = a_lo*b_lo + ((a_hi*b_lo + b_hi*a_lo) << `M);
endmodule

module testbench;
	rand reg [`N+`M-1:0] A, B;
	wire [`N+`M-1:0] D;

	top uut (A, B, D);

	always @* begin
		assert (D == 0);
	end
endmodule
