module ioshim_mem (
	input clk,
	input resetn,

	input [1:0] wen1,
	input [10:0] addr1, 
	input [15:0] wdata1, 
	output reg [15:0] rdata1, 

	input [10:0] addr2,
	output reg [15:0] rdata2
);
	parameter integer MEMSIZE = 128;
	parameter MEMINIT = "";

	reg [15:0] memory [0:MEMSIZE-1];

	initial if (MEMINIT)
		$readmemh(MEMINIT, memory);

	always @(posedge clk) begin
		if (wen1[1]) memory[addr1][15:8] <= wdata1[15:8];
		if (wen1[0]) memory[addr1][ 7:0] <= wdata1[ 7:0];
		rdata1 <= memory[addr1];
		rdata2 <= memory[addr2];
	end
endmodule
