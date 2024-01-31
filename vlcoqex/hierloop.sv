`define N 4

module top (
	input [`N-1:0] A, U, V,
	output [`N-1:0] Y
);
	wire [`N-1:0] X;
	add_sub add_sub_inst (A, X, U, V, X, Y);
endmodule

module add_sub (
	input [`N-1:0] A, B, U, V,
	output [`N-1:0] X, Y
);
	assign X = A + U, Y = B - V;
endmodule

module testbench;
	rand reg [`N-1:0] A, U, V;
	wire [`N-1:0] Y;

	top uut (A, U, V, Y);

	always @* begin
		assume (U == V);
		assert (Y == A);
	end
endmodule
