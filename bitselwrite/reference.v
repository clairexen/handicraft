module reference (
	input clk ,
	input [9:0] ctrl ,
	input [15:0] din ,
	input [3:0] sel ,
	output reg [1023:0] dout
);
	always @(posedge clk) begin
		dout[ctrl*sel+:64] <= din ;
	end
endmodule
