
`timescale 1ns / 1ps

module example_tb();

reg clk, rst;
wire [7:0] example_out;
reg [7:0] example_in;

example UUT (
	.clk(clk),
	.rst(rst),
	.out(example_out),
	.in(example_in)
);

always @(posedge clk) begin
	if (!rst && example_out != 0)
		$display("%t %c", $time, example_out);
end

initial begin
	clk <= 0;
	#100;
	forever begin
		clk <= !clk;
		#5;
	end
end

initial begin
	rst <= 1;
	#100;
	@(negedge clk);
	rst <= 0;
end

initial begin
	// $dumpfile("example_tb.vcd");
	// $dumpvars(0, example_tb);

	example_in <= 0;
	#200;
	@(negedge clk);
	example_in <= "x";
	@(negedge clk);
	example_in <= "y";
	@(negedge clk);
	example_in <= "z";
	@(negedge clk);
	example_in <= "\n";
	@(negedge clk);
	example_in <= 0;
	#200;
	$finish;
end

endmodule

