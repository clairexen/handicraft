module testbench;
	localparam [31:0] test_din = 32'b 10110011001110001001111000111001;
	localparam [31:0] test_msk = 32'b 01101001000010101110101001110101;
	localparam [31:0] test_sag = 32'b 01001100110010110101101001101101;
	localparam [31:0] test_isg = 32'b 01000011110011111000001111100011;

	reg  clock;
	reg  reset;

	reg  ctrl_inv;
	reg  ctrl_msk;
	reg  ctrl_ldm;

	reg  ctrl_start;
	wire ctrl_ready;

	reg  [31:0] in_data;
	reg  [31:0] in_mask;
	wire [31:0] out_data;

`ifdef TYPE_32
`ifdef TYPE_C
	assign ctrl_ready = 1;
	SAG4Fun32C uut (
		.ctrl_inv (ctrl_inv),
		.ctrl_msk (ctrl_msk),
		.in_data  (in_data ),
		.in_mask  (in_mask ),
		.out_data (out_data)
	);
`endif
`ifdef TYPE_S
	SAG4Fun32S uut (
		.clock (clock),
		.reset (reset),

		.ctrl_inv (ctrl_inv),
		.ctrl_msk (ctrl_msk),
		.ctrl_ldm (ctrl_ldm),

		.ctrl_start (ctrl_start),
		.ctrl_ready (ctrl_ready),

		.in_data  (in_data ),
		.out_data (out_data)
	);
`endif
`endif

	initial begin
		if ($test$plusargs("vcd")) begin
			$dumpfile("testbench.vcd");
			$dumpvars(0, testbench);
		end

		#5 clock = 0;
		forever #5 clock = ~clock;
	end


	initial begin
		$display("in_data  = %b", test_din);
		$display("in_mask  = %b", test_msk);

		reset <= 1;
		ctrl_inv <= 0;
		ctrl_msk <= 0;
		ctrl_ldm <= 1;
		ctrl_start <= 0;
		in_data <= test_msk;
		in_mask <= test_msk;
		@(posedge clock);

		reset <= 0;
		ctrl_start <= 1;
		@(posedge clock);

		ctrl_start <= 0;
`ifdef TYPE_S
		ctrl_inv <= 'bx;
		ctrl_msk <= 'bx;
		ctrl_ldm <= 'bx;
		in_data <= 'bx;
`endif
		@(posedge clock);
		@(posedge clock);
		@(posedge clock);
		@(posedge clock);

		ctrl_inv <= 0;
		ctrl_msk <= 0;
		ctrl_ldm <= 0;
		ctrl_start <= 1;
		in_data <= test_din;
		@(posedge clock);

		ctrl_start <= 0;
`ifdef TYPE_S
		ctrl_inv <= 'bx;
		ctrl_msk <= 'bx;
		ctrl_ldm <= 'bx;
		in_data <= 'bx;
`endif
		@(posedge clock);
		@(posedge clock);
		@(posedge clock);
		@(posedge clock);

		ctrl_inv <= 1;
		ctrl_msk <= 0;
		ctrl_ldm <= 0;
		ctrl_start <= 1;
		in_data <= test_din;

		#1 $display("SAG:");
		$display("expected = %b", test_sag);
		$display("out_data = %b %s", out_data, out_data === test_sag ? "OK" : "ERROR");
		@(posedge clock);

		ctrl_start <= 0;
`ifdef TYPE_S
		ctrl_inv <= 'bx;
		ctrl_msk <= 'bx;
		ctrl_ldm <= 'bx;
		in_data <= 'bx;
`endif
		@(posedge clock);
		@(posedge clock);
		@(posedge clock);
		@(posedge clock);

		#1 $display("ISG:");
		$display("expected = %b", test_isg);
		$display("out_data = %b %s", out_data, out_data === test_isg ? "OK" : "ERROR");
		@(posedge clock);

		@(negedge clock);
		#2 $finish;
	end
endmodule
