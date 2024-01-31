module lut #(
	parameter N_INPUTS = 3
) (
	input [2**N_INPUTS-1:0] cfg,
	input [N_INPUTS-1:0] din,
	output dout
);
	assign dout = cfg[din];
endmodule
