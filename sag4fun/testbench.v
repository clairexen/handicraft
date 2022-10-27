module testbench;
	reg  ctrl_inv;
	reg  ctrl_msk;
	reg  [31:0] in_data;
	reg  [31:0] in_mask;
	wire [31:0] out_data;

	SAG4Fun32c uut (
		.ctrl_inv(ctrl_inv),
		.ctrl_msk(ctrl_msk),
		.in_data (in_data ),
		.in_mask (in_mask ),
		.out_data(out_data)
	);

	initial begin
		ctrl_inv = 0;
		ctrl_msk = 0;
		in_data = 32'b 10110011001110001001111000111001;
		in_mask = 32'b 01101001000010101110101001110101;
		#1 $display("%b %s", out_data, out_data == 32'b 01001100110010110101101001101101 ? "OK" : "ERROR");
	end
endmodule
