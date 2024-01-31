module improved2 (
	input clk ,
	input [9:0] ctrl ,
	input [15:0] din ,
	input [3:0] sel ,
	output reg [1023:0] dout
);
	wire [1023:0] dout_mask = 64'h FFFF_FFFF_FFFF_FFFF << ctrl*sel;
	wire [63:0] dout_data = ({48'b0, din, 48'b0, din} << ({ctrl*sel} & 63)) >> 64;
	always @(posedge clk) begin
		dout <= (dout & ~dout_mask) | ({1024/64{dout_data}} & dout_mask);
	end
endmodule
