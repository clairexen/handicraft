module top (clk, rst, din, dout, wr_en, rd_en, rd_empty, error, half_full);
	input clk, rst;
	input [33:0] din;
	output [38:0] dout;
	input wr_en, rd_en;
	output rd_empty, error, half_full;

	wxzip wxzip_core (
		.clk(clk),
		.rst(rst),
		.din(din),
		.dout(dout),
		.wr_en(wr_en),
		.rd_en(rd_en),
		.rd_empty(rd_empty),
		.error(error),
		.half_full(half_full)
	);
endmodule

// blackbox declaration
module wxzip (clk, rst, din, dout, wr_en, rd_en, rd_empty, error, half_full);
	input clk, rst;
	input [33:0] din;
	output [38:0] dout;
	input wr_en, rd_en;
	output rd_empty, error, half_full;
endmodule
