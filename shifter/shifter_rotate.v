module shifter_rotate (input [31:0] H, L, input [4:0] shamt, input reverse, output [31:0] Y);
	wire [5:0] real_shamt = reverse ? -shamt : shamt;
	assign Y = {L, H, L} >> real_shamt;
endmodule
