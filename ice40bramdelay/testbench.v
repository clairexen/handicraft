module testbench;
	reg clk = 1;
	always #5 clk = ~clk;

	wire [7:0] leds;

	test uut (
		.clk(clk),
		.leds(leds)
	);

	initial begin
		$dumpfile("testbench.vcd");
		$dumpvars(0, testbench);
		repeat (10000) @(posedge clk);
		$finish;
	end
endmodule
