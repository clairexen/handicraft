
module array_multiplier_pipeline_tb;

reg clk;
reg [31:0] a, b;
wire [31:0] y;

array_multiplier_pipeline #(
	.width(32)
) uut (
	.clk(clk),
	.a(a),
	.b(b),
	.y(y)
); 

reg [31:0] xrnd = 1;
task xorshift32;
begin
	xrnd = xrnd ^ (xrnd << 13);
	xrnd = xrnd ^ (xrnd >> 17);
	xrnd = xrnd ^ (xrnd <<  5);
end endtask

initial begin
	clk <= 0;
	forever begin
		#10;
		clk <= ~clk;
	end
end

integer i, j;
reg [31:0] backup_a, backup_b;

initial begin
	// $dumpfile("array_multiplier_pipeline_tb.vcd");
	// $dumpvars(0, array_multiplier_pipeline_tb);

	@(posedge clk);
	@(posedge clk);

	for (i = 0; i < 100; i = i+1)
	begin
		xorshift32;
		a <= xrnd;
		backup_a <= xrnd;

		xorshift32;
		b <= xrnd;
		backup_b <= xrnd;

		@(posedge clk);

		a <= 32'bx;
		b <= 32'bx;

		for (j = 0; j < 32; j = j+1)
			@(posedge clk);

		$display("%d * %d = %d (%d)", backup_a, backup_b, y, backup_a*backup_b);
		if (!(y === backup_a*backup_b)) begin
			$display("ERROR!");
			$finish;
		end
	end

	$display("PASSED.");
	$finish;
end

endmodule

