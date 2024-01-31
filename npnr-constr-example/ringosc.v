`timescale 1ns / 1ps

module top(output OUT);

	localparam DIV_POW2 = 20;

	wire [DIV_POW2:0] clkdiv;
	genvar i;

	// ring oscillator

	SB_LUT4 #(
		.LUT_INIT(16'b 0000_0000_0000_0001)
	) inv (
		.O(clkdiv[0]),
		.I0(clkdiv[0]),
		.I1(1'b0),
		.I2(1'b0),
		.I3(1'b0)
	);

	// clock divider

	generate for (i = 0; i < DIV_POW2; i = i+1) begin:clkdiv_stage
		reg out;
		wire in = clkdiv[i];
		assign clkdiv[i+1] = out;
		always @(posedge in) out <= !out;
	end endgenerate

	assign OUT = clkdiv[DIV_POW2];
endmodule
