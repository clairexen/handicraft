`timescale 1 ns / 1 ps

module top(input [2:0] a, output [3:0] y);
	localparam [5:15] p = 51681708;
	assign y = p[15 + a -: 5];
endmodule

