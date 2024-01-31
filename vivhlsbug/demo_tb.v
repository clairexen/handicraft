`timescale 1 ns / 1 ps

module testbench;
	reg clk = 1;
	always #50 clk = ~clk;

	reg         ap_rst_n = 0;

	reg         in_stream_V_TVALID = 0;
	wire        in_stream_V_TREADY;
	reg  [31:0] in_stream_V_TDATA;

	wire        out_stream_V_TVALID;
	reg         out_stream_V_TREADY = 0;
	wire [31:0] out_stream_V_TDATA;

	reg [32:0] testdata [0:299];
	initial $readmemh("demo_tb.dat", testdata);

	reg got_error = 0;
	integer in_idx, out_idx;

	initial begin
		repeat (100) @(posedge clk);
		ap_rst_n <= 1;
		repeat (100) @(posedge clk);

		for (in_idx = 0; in_idx < 300; in_idx = in_idx+1) begin
			if (testdata[in_idx][32] == 0) begin
				in_stream_V_TVALID <= 1;
				in_stream_V_TDATA <= testdata[in_idx][31:0];
				@(posedge clk);
				while (!in_stream_V_TREADY) @(posedge clk);
			end
		end

		in_stream_V_TVALID <= 0;
		in_stream_V_TDATA <= 'bx;
	end

	initial begin
		repeat (200) @(posedge clk);
		out_stream_V_TREADY <= 1;

		for (out_idx = 0; out_idx < 300; out_idx = out_idx+1) begin
			if (testdata[out_idx][32] == 1) begin
				while (!out_stream_V_TVALID) @(posedge clk);
				if (out_stream_V_TDATA == testdata[out_idx][31:0]) begin
					$display("out=%x ref=%x OK", out_stream_V_TDATA, testdata[out_idx][31:0]);
				end else begin
					$display("out=%x ref=%x ERROR", out_stream_V_TDATA, testdata[out_idx][31:0]);
					got_error <= 1;
				end
				@(posedge clk);
			end
		end

		repeat (100) @(posedge clk);

		if (got_error) begin
			$display("FAIL");
			$stop;
		end else begin
			$display("PASS");
			$finish;
		end
	end

	initial begin
		repeat (1000) @(posedge clk);
		$display("Timeout!");
		$display("FAIL");
		$stop;
	end

	demo uut (
		.ap_clk             (clk                ),
		.ap_rst_n           (ap_rst_n           ),

		.in_stream_V_TVALID (in_stream_V_TVALID ),
		.in_stream_V_TREADY (in_stream_V_TREADY ),
		.in_stream_V_TDATA  (in_stream_V_TDATA  ),

		.out_stream_V_TVALID(out_stream_V_TVALID),
		.out_stream_V_TREADY(out_stream_V_TREADY),
		.out_stream_V_TDATA (out_stream_V_TDATA )
	);
endmodule
