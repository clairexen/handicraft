
`timescale 1 ns / 1 ps

module testbench;
	reg        ap_clk = 0;
	reg        ap_rst = 1;
	reg        ap_start = 0;
	wire       ap_done;
	wire       ap_idle;
	wire       ap_ready;
	reg  [4:0] index_V;
	wire [6:0] ap_return;

	always #5 ap_clk = ~ap_clk;

	initial begin
		$dumpfile("hlsbugtst5.vcd");
		$dumpvars(0, testbench);

		repeat (100) @(posedge ap_clk);
		ap_rst <= 0;

		repeat (1000) @(posedge ap_clk);
		$display("TIMEOUT!");
		$finish;
	end

	hlsbugtst5 uut (
		.ap_clk   (ap_clk   ),
		.ap_rst   (ap_rst   ),
		.ap_start (ap_start ),
		.ap_done  (ap_done  ),
		.ap_idle  (ap_idle  ),
		.ap_ready (ap_ready ),
		.index_V  (index_V  ),
		.ap_return(ap_return)
	);

	integer i;

	initial begin
		while (ap_rst) @(posedge ap_clk);
		while (!ap_idle) @(posedge ap_clk);

		for (i = 0; i < 32; i=i+1) begin
			repeat (10) @(posedge ap_clk);
			index_V <= i;
			ap_start <= 1;

			@(posedge ap_clk);
			ap_start <= 0;

			while (!ap_idle) @(posedge ap_clk);
			$display("%2d %3d", i, ap_return);
		end

		$finish;
	end
endmodule
