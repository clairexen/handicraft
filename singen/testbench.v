module testbench;
	reg clk = 1;
	always #5 clk = ~clk;

	singen uut (
		.clk(clk),
		.out_sq(out_sq),
		.out_pwm(out_pwm),
		.out_dm(out_dm)
	);

	initial begin
		$dumpfile("testbench.vcd");
		$dumpvars(0, testbench);
		repeat (100000) @(posedge clk);
		$finish;
	end
endmodule
