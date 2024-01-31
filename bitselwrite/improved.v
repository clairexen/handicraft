module improved (
	input clk ,
	input [9:0] ctrl ,
	input [15:0] din ,
	input [3:0] sel ,
	output reg [1023:0] dout
);
	wire [1023:0] dout_mask = 64'h FFFF_FFFF_FFFF_FFFF << ctrl*sel;
	wire [1023:0] dout_data = din << ctrl*sel;
	always @(posedge clk) begin
		dout <= (dout & ~dout_mask) | dout_data;
	end
endmodule
