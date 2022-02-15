module top (
	input clock,
	input [11:0] ctrl
);
	reg [3:0] state = 0;
	always @(posedge clock) begin
		if (ctrl[state[3:1]] == state[0])
			state <= state + 1;
		else
			state <= 0;
	end
	always_comb assert (state != 15);
endmodule
