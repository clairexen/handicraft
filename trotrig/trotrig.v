// clifford's trojan trigger
module trotrig #(
	parameter DINBITS = 8,
	parameter COUNTBITS = 20
) (
	input clk, reset, enable,
	input [DINBITS-1:0] din,
	output reg trigger
);
	reg [31:0] state, next_state;

	always @* begin
		next_state = state;
		next_state[31:16] = next_state[31:16] ^ next_state[15:0];
		next_state[15: 8] = next_state[15: 8] ^ next_state[ 7:0];
		next_state = reset ? 32'd 0 : next_state + din + (32'd 1 << DINBITS);
		trigger = next_state > (32'd 1 << (DINBITS+COUNTBITS));
		next_state[15: 8] = next_state[15: 8] ^ next_state[ 7:0];
		next_state[31:16] = next_state[31:16] ^ next_state[15:0];
	end

	always @(posedge clk) begin
		if (enable || reset)
			state <= next_state;
	end
endmodule
