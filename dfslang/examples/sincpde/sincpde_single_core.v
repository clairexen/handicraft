
`timescale  1 ns / 1 ps

module sincpde(rst, clk, sync_in, sync_out, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, p, ampl);

input rst, clk;
output sync_in, sync_out;

input [17:0] s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10;
output [17:0] p, ampl;

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

wire SUBAVG_0_mode;
wire signed [17:0] SUBAVG_0_a;
wire signed [17:0] SUBAVG_0_b;
reg signed [17:0] SUBAVG_0_y;

wire signed [47:0] SHIFT_0_a;
reg signed [47:0] SHIFT_0_shr1;
reg signed [17:0] SHIFT_0_shr19;

wire signed [47:0] STEP_0_x;
wire STEP_0_d;
reg signed [47:0] STEP_0_y;

DFS_CORE uut (
	.clk(clk),
	.rst(rst),
	.sync_in(sync_in),
	.sync_out(sync_out),
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
	.SHIFT_0_a(SHIFT_0_a),
	.STEP_0_d(STEP_0_d),
	.STEP_0_x(STEP_0_x),
	.SUBAVG_0_a(SUBAVG_0_a),
	.SUBAVG_0_b(SUBAVG_0_b),
	.SUBAVG_0_mode(SUBAVG_0_mode),
	.sample0(s0),
	.sample1(s1),
	.sample10(s10),
	.sample2(s2),
	.sample3(s3),
	.sample4(s4),
	.sample5(s5),
	.sample6(s6),
	.sample7(s7),
	.sample8(s8),
	.sample9(s9),
	.DIV_0_Q(DIV_0_Q),
	.IMAC_0_Z(IMAC_0_Z),
	.SHIFT_0_shr1(SHIFT_0_shr1),
	.SHIFT_0_shr19(SHIFT_0_shr19),
	.STEP_0_y(STEP_0_y),
	.SUBAVG_0_y(SUBAVG_0_y),
	.ampl(ampl),
	.p(p)
);

SINCPDE_DIV div_0 (clk, DIV_0_N <<< 2, DIV_0_D, DIV_0_Q, DIV_0_sync_in, DIV_0_sync_out_ref);

SINCPDE_IMAC imac_0 (clk, IMAC_0_sync_in, IMAC_0_sync_out_ref, IMAC_0_X, -IMAC_0_Y, IMAC_0_MODE,
		{IMAC_0_S0, IMAC_0_S1, IMAC_0_S2, IMAC_0_S3, IMAC_0_S4, IMAC_0_S5,
		IMAC_0_S6, IMAC_0_S7, IMAC_0_S8, IMAC_0_S9, IMAC_0_S10}, IMAC_0_Z);

always @(posedge clk) begin
	SUBAVG_0_y <= SUBAVG_0_mode == 0 ? SUBAVG_0_a - SUBAVG_0_b : (SUBAVG_0_a + SUBAVG_0_b + 19'sd0) / 2;
	SHIFT_0_shr1 <= SHIFT_0_a >>> 1;
	SHIFT_0_shr19 <= SHIFT_0_a >>> 19;
	STEP_0_y <= STEP_0_x + (STEP_0_d ? +(1 << 14) : -(1 << 14));
end

endmodule
