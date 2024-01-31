
module testbench;

reg rst, clk;
wire sync_in, sync_out;
reg signed [15:0] data_in;
wire signed [15:0] data_out;

wire signed [15:0] MACC3_0_A;
wire signed [15:0] MACC3_0_B;
wire signed [15:0] MACC3_0_C;
wire signed [15:0] MACC3_0_Y;
wire MACC3_0_clear, MACC3_0_sub;

DFS_CORE dfs_core (
	.clk(clk),
	.rst(rst),
	.sync_in(sync_in),
	.sync_out(sync_out),
	.MACC3_0_Y(MACC3_0_Y),
	.in(data_in),
	.MACC3_0_A(MACC3_0_A),
	.MACC3_0_B(MACC3_0_B),
	.MACC3_0_C(MACC3_0_C),
	.MACC3_0_clear(MACC3_0_clear),
	.MACC3_0_sub(MACC3_0_sub),
	.out(data_out)
);

MACC3 macc (clk, MACC3_0_A, MACC3_0_B, MACC3_0_C, MACC3_0_clear, MACC3_0_sub, MACC3_0_Y);

initial begin
	rst <= 1;
	#100;
	@(posedge clk);
	@(posedge clk);
	@(posedge clk);
	rst <= 0;
end

initial begin
	clk <= 0;
	forever begin
		#10;
		clk <= ~clk;
	end
end

initial begin
	@(posedge clk);
	@(posedge clk);
	@(posedge clk);

	data_in <= 4; while (!sync_in) @(posedge clk);
	data_in <= 7; @(posedge clk);
	data_in <= 5; @(posedge clk);
	data_in <= 2; @(posedge clk);
	data_in <= 6; @(posedge clk);
	data_in <= 4; @(posedge clk);
	data_in <= 9; @(posedge clk);
	data_in <= 2; @(posedge clk);
	data_in <= 1; @(posedge clk);

	while (!sync_out) @(posedge clk);
	$display("det_1: %d", data_out); @(posedge clk); // expected:  -20
	$display("det_2: %d", data_out); @(posedge clk); // expected:   20
	$display("det_M: %d", data_out); @(posedge clk); // expected: -400

	while (!sync_out) @(posedge clk);
	$display("det_1: %d", data_out); @(posedge clk); // expected: 0
	$display("det_2: %d", data_out); @(posedge clk); // expected: 0
	$display("det_M: %d", data_out); @(posedge clk); // expected: 0

	$finish;
end

initial begin
	$dumpfile("det3_tb.vcd");
	$dumpvars(0, testbench);
end

endmodule

// ---------------------------------------------------------------------------------------

module MACC3 (clk, A, B, C, clear, sub, Y);

input signed [15:0] A;
input signed [15:0] B;
input signed [15:0] C;
input clk, clear, sub;

reg signed [15:0] Y_buf;
output reg signed [15:0] Y;

wire [15:0] prod;
assign prod = A*B*C;

always @(posedge clk) begin
	Y_buf <= (clear ? 15'd0 : Y_buf) + (sub ? -prod : +prod);
	Y <= Y_buf;
end

endmodule

