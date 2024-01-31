`timescale 1 ns / 1 ps

module testbench;
	reg [2:0] a;
	wire [3:0] y;

	top uut (
		.a(a),
		.y(y)
	);

	initial begin
		a = 3'b 001;
		#1000;
		$display("a=%b, y=%b", a, y);
		$finish;
	end
endmodule
