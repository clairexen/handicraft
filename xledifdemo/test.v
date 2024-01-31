module top (
	input clk, reset,
	input [7:0] a, b, c,
	output reg [7:0] y
);
	always @(posedge clk) begin
		if (reset) begin
			y <= 0;
		end else begin
			y <= ((a + b) ^ c) - y;
		end
	end
endmodule
