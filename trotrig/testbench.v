module testbench;
	reg clk = 1;
	always #5 clk = ~clk;

	reg reset = 1;
	reg enable = 0;
	reg [7:0] din;
	wire trigger;

	initial begin
		$dumpfile("testbench.vcd");
		$dumpvars(0, testbench);
		repeat (5)
			@(posedge clk);
		reset <= 0;
		@(posedge trigger);
		repeat (5)
			@(posedge clk);
		$finish;
	end

	always @(posedge clk) begin
		din <= $random;
		enable <= $random;
	end

	trotrig #(
		.DINBITS(8),
		.COUNTBITS(8)
	) uut (
		.clk(clk),
		.reset(reset),
		.enable(enable),
		.din(din),
		.trigger(trigger)
	);
endmodule
