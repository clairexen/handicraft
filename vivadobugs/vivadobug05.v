`timescale 1 ns / 1 ps

module vivadobug05(input clk, output led0, led1);
	reg [24:0] counter = 0;

	always @(posedge clk) begin
		counter <= counter + 1;
	end

	assign led1 = counter[24];
	assign led0 = counter[23];
endmodule
