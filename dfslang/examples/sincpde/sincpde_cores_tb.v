
`timescale  1 ns / 1 ps

module testbench;

reg rst, clk;
wire sync_in, sync_out;
wire signed [47:0] out;

wire DIV_0_sync_in, DIV_0_sync_out, DIV_0_sync_out_ref;
wire signed [47:0] DIV_0_N;
wire signed [47:0] DIV_0_D;
wire signed [17:0] DIV_0_Q;

wire IMAC_0_sync_in, IMAC_0_sync_out, IMAC_0_sync_out_ref;
wire [1:0] IMAC_0_MODE;
wire [17:0] IMAC_0_S0, IMAC_0_S1, IMAC_0_S2, IMAC_0_S3, IMAC_0_S4, IMAC_0_S5;
wire [17:0] IMAC_0_S6, IMAC_0_S7, IMAC_0_S8, IMAC_0_S9, IMAC_0_S10;
wire [17:0] IMAC_0_X;
wire [47:0] IMAC_0_Y;
wire [47:0] IMAC_0_Z;

DFS_CORE uut (
	.clk(clk),
	.rst(rst),
	.sync_in(sync_in),
	.sync_out(sync_out),
	.DIV_0_Q(DIV_0_Q),
	.IMAC_0_Z(IMAC_0_Z),
	.DIV_0_D(DIV_0_D),
	.DIV_0_N(DIV_0_N),
	.DIV_0_sync_in(DIV_0_sync_in),
	.DIV_0_sync_out(DIV_0_sync_out),
	.IMAC_0_MODE(IMAC_0_MODE),
	.IMAC_0_S0(IMAC_0_S0),
	.IMAC_0_S1(IMAC_0_S1),
	.IMAC_0_S10(IMAC_0_S10),
	.IMAC_0_S2(IMAC_0_S2),
	.IMAC_0_S3(IMAC_0_S3),
	.IMAC_0_S4(IMAC_0_S4),
	.IMAC_0_S5(IMAC_0_S5),
	.IMAC_0_S6(IMAC_0_S6),
	.IMAC_0_S7(IMAC_0_S7),
	.IMAC_0_S8(IMAC_0_S8),
	.IMAC_0_S9(IMAC_0_S9),
	.IMAC_0_X(IMAC_0_X),
	.IMAC_0_Y(IMAC_0_Y),
	.IMAC_0_sync_in(IMAC_0_sync_in),
	.IMAC_0_sync_out(IMAC_0_sync_out),
	.out(out)
);

SINCPDE_DIV div (clk, DIV_0_N, DIV_0_D, DIV_0_Q, DIV_0_sync_in, DIV_0_sync_out_ref);
SINCPDE_IMAC imac (clk, IMAC_0_sync_in, IMAC_0_sync_out_ref, IMAC_0_X, IMAC_0_Y, IMAC_0_MODE,
		{IMAC_0_S0, IMAC_0_S1, IMAC_0_S2, IMAC_0_S3, IMAC_0_S4, IMAC_0_S5,
		IMAC_0_S6, IMAC_0_S7, IMAC_0_S8, IMAC_0_S9, IMAC_0_S10}, IMAC_0_Z);

always @(posedge clk) begin
	if (DIV_0_sync_out_ref && DIV_0_sync_out !== DIV_0_sync_out_ref)
		$display("divider sync failed: state=%d", uut.state);
end

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

	while (!sync_out) @(posedge clk);
	$display("div out:  %d", out); @(posedge clk);
	$display("div out:  %d", out); @(posedge clk);
	$display("div out:  %d", out); @(posedge clk);
	$display("div out:  %d", out); @(posedge clk);
	$display("div out:  %d", out); @(posedge clk);
	$display("div out:  %d", out); @(posedge clk);
	$display("div out:  %d", out); @(posedge clk);
	$display("imac out: %d", out); @(posedge clk);
	$display("imac out: %d", out); @(posedge clk);
	$display("imac out: %d", out); @(posedge clk);
	$display("imac out: %d", out); @(posedge clk);
	$display("imac out: %d", out); @(posedge clk);
	$display("imac out: %d", out); @(posedge clk);

	$finish;
end

initial begin
	$dumpfile("sincpde_cores_tb.vcd");
	$dumpvars(0, testbench);
end
endmodule

