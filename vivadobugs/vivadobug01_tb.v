`timescale 1 ns / 1 ps

module testbench;
	reg s;
	reg [1:0] x;
	wire [1:0] y;

	top uut (
		.s(s),
		.x(x),
		.y(y)
	);

	initial begin
		s = 1'b 1;
		x = 2'b 11;
		#1000;
		$display("s=%b, x=%b, y=%b", s, x, y);
		$finish;
	end
endmodule
