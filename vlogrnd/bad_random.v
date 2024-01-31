// see http://electronics.stackexchange.com/questions/96573/unexpected-patterns-in-verilog-random

module testbench;

integer i, j = 0;
reg [31:0] randval;

initial begin
	for (i = 0; i < 100000; i = i+1) begin
		randval = $random;
		if (randval[0 +: 6] == 0) begin
			$display("%8d %6d: %x -> %d %d", i, j, randval, randval[0 +: 6], randval[24 +: 4]);
			if (randval[24 +: 4] != 0) $finish;
			j = j + 1;
		end
	end
	$display("IEEE Std 1364-2005 $random (rtl_dist_uniform(seed, LONG_MIN, LONG_MAX)) sucks!");
end

endmodule
