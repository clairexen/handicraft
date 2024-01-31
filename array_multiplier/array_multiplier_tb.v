
module array_multiplier_tb;

reg [31:0] a, b;
wire [31:0] y;

array_multiplier #(
	.width(32)
) uut (
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

integer i;
initial begin
	// $dumpfile("array_multiplier_tb.vcd");
	// $dumpvars(0, array_multiplier_tb);

	for (i = 0; i < 100; i = i+1)
	begin
		#10;

		xorshift32;
		a <= xrnd;
		xorshift32;
		b <= xrnd;

		#10;

		$display("%d * %d = %d (%d)", a, b, y, a*b);
		if (y != a*b) begin
			$display("ERROR!");
			$finish;
		end
	end
	$display("PASSED.");
end

endmodule

