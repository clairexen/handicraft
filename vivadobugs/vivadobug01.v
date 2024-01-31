`timescale 1 ns / 1 ps

module top(input s, input [1:0] x, output [1:0] y);
	wire u = 1'bx;
	assign y = s ? u : x;
	// assign y = s ? 2'b0x : x;
endmodule
