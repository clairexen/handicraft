`timescale 1ns/1ns

module top_tb;
	reg clk_in = 0;
	always #5 clk_in = !clk_in;

	wire [4:0] led;
	wire [7:0] debug;
	integer i;

	`ifdef POST_SYN
	top t(
	    .sysclk    ( clk_in ),
	    .\led[0]   ( led[0] ),
	    .\led[1]   ( led[1] ),
	    .\led[2]   ( led[2] ),
	    .\led[3]   ( led[3] ),
	    .\led[4]   ( led[4] ),
	    .\debug[0] ( debug[0] ),
	    .\debug[1] ( debug[1] ),
	    .\debug[2] ( debug[2] ),
	    .\debug[3] ( debug[3] ),
	    .\debug[4] ( debug[4] ),
	    .\debug[5] ( debug[5] ),
	    .\debug[6] ( debug[6] ),
	    .\debug[7] ( debug[7] )
	);
	`else
	top t(
	    .sysclk  ( clk_in ),
	    .led     ( led ),
	    .debug   ( debug )
	);
	`endif

	initial begin
		$dumpfile(`VCD_FILE);
		$dumpvars(1, led);

		for (i = 0; i < 2000; i = i+1) begin
			repeat (10000) @(posedge clk_in);
			$write("%ccycle %d, %.2f%%", 13, i*10000, $itor(i)*100.0/2000.0);
			$fflush;
		end
		$display("");
		$finish;
	end
endmodule
