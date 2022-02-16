module top (
	input clock,
	input [7:0] ctrl
);
	reg init = 1;
	reg [3:0] state;
	always @(posedge clock) begin
		if ((ctrl & state) != state || init)
			state <= 1;
		else
			state <= state << 1;
		init <= 0;
	end
	always_comb assert (init || state != 4'b 1000);
endmodule
