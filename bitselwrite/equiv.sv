module miter (
	input clk ,
	input [9:0] ctrl ,
	input [15:0] din ,
	input [3:0] sel
);
	wire [1023:0] dout0, dout1, dout2;

`ifdef CHECK_REFERENCE
	reference reference_inst (
		.clk (clk  ),
		.ctrl(ctrl ),
		.din (din  ),
		.sel (sel  ),
		.dout(dout0)
	);
`else
	assign dout0 == dout1;
`endif

	improved improved_inst (
		.clk (clk  ),
		.ctrl(ctrl ),
		.din (din  ),
		.sel (sel  ),
		.dout(dout1)
	);

	improved2 improved2_inst (
		.clk (clk  ),
		.ctrl(ctrl ),
		.din (din  ),
		.sel (sel  ),
		.dout(dout2)
	);

	initial assume (dout0 == dout1);
	initial assume (dout1 == dout2);
	always @* assert (dout0 == dout1);
	always @* assert (dout1 == dout2);
endmodule
