module serial_alu_tb;
	reg        clock;
	reg        reset;

	initial begin
		$dumpfile("serial_alu_tb.vcd");
		$dumpvars(0, serial_alu_tb);

		#5 clock = 0;
		repeat (100000)
			#5 clock = ~clock;

		#10;
		$display("PASS");
		$finish;
	end

	always begin
		reset <= 1;
		@(posedge clock);

		reset <= 0;
		repeat (30) @(posedge clock); // 30 is too small, change to 50 to catch the bug in the UUT
	end

	reg  [7:0] din_di1;
	reg  [7:0] din_di2;
	reg  [1:0] din_fun;
	reg        din_vld;
	wire       din_rdy;

	wire [7:0] dout_dat;
	wire       dout_vld;

	serial_alu uut (
		.clock   (clock   ),
		.reset   (reset   ),
		.din_di1 (din_di1 ),
		.din_di2 (din_di2 ),
		.din_fun (din_fun ),
		.din_vld (din_vld ),
		.din_rdy (din_rdy ),
		.dout_dat(dout_dat),
		.dout_vld(dout_vld)
	);

	reg [7:0] expected_dout;

	always @(posedge clock) begin
		if (reset) begin
			expected_dout <= $random;
		end else begin
			if (din_vld && din_rdy) begin
				case (din_fun)
					0: expected_dout <= din_di1 + din_di2;
					1: expected_dout <= din_di1 & din_di2;
					2: expected_dout <= din_di1 | din_di2;
					3: expected_dout <= din_di1 ^ din_di2;
				endcase
			end
			if (dout_vld) begin
				// coverage off
				if (expected_dout !== dout_dat) begin
					$display("ERROR incorrect data out");
					$stop;
				end
				// coverage on
			end
		end
	end

/**
	// additional tests for testing vld/rdy timings

	reg [3:0] countdown;

	always @(posedge clock) begin
		if (reset) begin
			countdown <= 0;
		end else begin
			// coverage off
			if (dout_vld !== (countdown == 1)) begin
				$display("ERROR incorrect dout_vld timing");
				$stop;
			end
			if (din_rdy !== (countdown <= 1)) begin
				$display("ERROR incorrect din_rdy timing");
				$stop;
			end
			// coverage on
			if (countdown) begin
				countdown <= countdown-1;
			end
			if (din_vld && din_rdy) begin
				countdown <= 9;
			end
		end
	end
**/

	always @(posedge clock) begin
		if (reset) begin
			din_di1 <= ($random & 3) ? $random : 0;
			din_di2 <= ($random & 3) ? $random : 0;
			din_fun <= 0;
		end else
		if (din_vld && din_rdy) begin
			din_di1 <= ($random & 3) ? $random : 0;
			din_di2 <= ($random & 3) ? $random : 0;
			din_fun <= din_fun + 1;
		end
		din_vld <= $random;
	end
endmodule
