`timescale 1 ns / 1 ps

module testbench;
	wire [7:0] a, b;

	top uut (
		.a(a),
		.b(b)
	);

	initial begin
		#1000;
		$display("a=%s, b=%s", a, b);
		$finish;
	end
endmodule
