module top (
	input clock,
	input [3:0] ctrl
);
	reg init = 1;
	reg [4:0] state;
	always @(posedge clock) begin
		if ((ctrl & state) != state || init)
			state <= 1;
		else
			state <= state << 1;
		init <= 0;
	end
	always_comb assert (init || state != 5'b 10000);
endmodule
