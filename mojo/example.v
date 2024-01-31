module top(clk, led);

input clk;
output [7:0] led;

reg [31:0] counter;

always @(posedge clk)
	counter <= counter + 1;

assign led = counter >> 24;

endmodule
