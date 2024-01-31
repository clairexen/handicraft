module test1 (
	input i_clk,
	input o_wb_stb,
	input i_wb_stall,
	input i_wb_we,
	input [7:0] i_wb_addr
);
	reg f_past_valid;

	initial f_past_valid = 1'b0;

	always @(posedge i_clk)
		f_past_valid <= 1'b1;
	
	always @(posedge i_clk)
		if ((f_past_valid) && ($past(o_wb_stb)) && ($past(i_wb_stall)))
			assert ((o_wb_stb) && ($stable({i_wb_addr, i_wb_we})));
endmodule

module test2 (
	input i_clk,
	input o_wb_stb,
	input i_wb_stall,
	input i_wb_we,
	input [7:0] i_wb_addr
);
	assert property (@(posedge i_clk)
		(o_wb_stb) && (i_wb_stall) |=> o_wb_stb && ($stable({i_wb_addr, i_wb_we})));
endmodule
