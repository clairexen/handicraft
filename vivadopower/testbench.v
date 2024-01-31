module testbench;
	reg clk = 1;
	reg resetn = 0;

	always #5 clk = ~clk;

	reg         pcpi_valid;
	reg  [31:0] pcpi_insn;
	reg  [31:0] pcpi_rs1;
	reg  [31:0] pcpi_rs2;

	wire        pcpi_wr;
	wire [31:0] pcpi_rd;
	wire        pcpi_wait;
	wire        pcpi_ready;

	reg [31:0] x32 = 314159265;

	task xorshift32; begin
		x32 = x32 ^ (x32 << 13);
		x32 = x32 ^ (x32 >> 17);
		x32 = x32 ^ (x32 << 5);
	end endtask

	initial begin
		pcpi_valid <= 0;

		repeat (100) @(posedge clk);
		resetn <= 1;

		repeat (100) begin
			repeat (10) @(posedge clk);
			pcpi_valid <= 1;

			xorshift32;
			pcpi_insn <= x32;
			pcpi_insn[31:25] <= 1;
			pcpi_insn[14] <= 0;
			pcpi_insn[6:0] <= 7'b0110011;

			xorshift32;
			pcpi_rs1 <= x32;

			xorshift32;
			pcpi_rs2 <= x32;

			@(posedge clk);
			while (!pcpi_ready) @(posedge clk);
			pcpi_valid <= 0;
		end

		repeat (10) @(posedge clk);
		$finish;
	end

	demo uut (
		.clk       (clk       ),
		.resetn    (resetn    ),
		.pcpi_valid(pcpi_valid),
		.pcpi_insn (pcpi_insn ),
		.pcpi_rs1  (pcpi_rs1  ),
		.pcpi_rs2  (pcpi_rs2  ),
		.pcpi_wr   (pcpi_wr   ),
		.pcpi_rd   (pcpi_rd   ),
		.pcpi_wait (pcpi_wait ),
		.pcpi_ready(pcpi_ready)
	);
endmodule
