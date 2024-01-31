module testbench;
	reg clock, reset = 1;
	always #5 clock = (clock === 1'b0);
	always @(posedge clock) reset <= 0;

	initial begin
		$dumpfile("testbench.vcd");
		$dumpvars(0, testbench);
		repeat (60000) @(posedge clock);
		$finish;
	end

	wire [7:0] out;
	fizzbuzz uut (
		.clock(clock),
		.reset(reset),
		.out(out)
	);

	always @(posedge clock)
		if (!reset) $write("%c", out);
endmodule
