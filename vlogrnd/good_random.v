// see http://electronics.stackexchange.com/questions/96573/unexpected-patterns-in-verilog-random

module testbench;

reg [63:0] xorshift64_state = 64'd88172645463325252;

task xorshift64_next;
	begin
		// see page 4 of Marsaglia, George (July 2003). "Xorshift RNGs". Journal of Statistical Software 8 (14).
		xorshift64_state = xorshift64_state ^ (xorshift64_state << 13);
		xorshift64_state = xorshift64_state ^ (xorshift64_state >>  7);
		xorshift64_state = xorshift64_state ^ (xorshift64_state << 17);
	end
endtask

integer i;
initial begin
	for (i = 0; i < 100; i = i+1) begin
		xorshift64_next;
		$display("%x", xorshift64_state[31:0]);
	end
end

endmodule
