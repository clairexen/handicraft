module ioshim_regs (
	input clk,
	input resetn,

	input wr_en,
	input [3:0] wr_addr, 
	input [7:0] wr_data, 

	input [3:0] rd1_addr,
	output [7:0] rd1_data,

	input [3:0] rd2_addr,
	output [7:0] rd2_data
);
	reg [7:0] memory [0:15];

	assign rd1_data = (wr_en && (wr_addr == rd1_addr)) ? wr_data : memory[rd1_addr];
	assign rd2_data = (wr_en && (wr_addr == rd2_addr)) ? wr_data : memory[rd2_addr];

	always @(posedge clk) begin
		if (wr_en) begin
			memory[wr_addr] <= wr_data;
			// $display("Register write: 0x%02x => reg%1d", wr_addr, wr_data);
		end
	end
endmodule
