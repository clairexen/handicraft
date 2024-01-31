`timescale 1 ns / 1 ps

module top(input [2:0] a, output y);
	assign y = a != 1'bx;
endmodule

