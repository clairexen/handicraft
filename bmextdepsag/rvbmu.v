module rvbmu_generic_cell (
	input        OM = '0,   // Or-combine-Mode

	input  [1:0] DH = '0,   // Data-in-Hi
	input  [1:0] DL = '0,   // Data-in-Lo
	output [1:0] DO,        // Data-Out

	input        CI = '0,   // Carry-In
	input        LI = '0,   // rotate-Left-In
	input        RI = '0,   // rotate-Right-In
	input        BI = '0,   // control-Bit-In

	output       CO,        // Carry-Out
	output       BO         // control-Bit-Out
);
	wire [1:0] di = !LI && RI ? DH : LI && !RI ? DL : {DH[0],DL[1]};

	assign DO = BI ? (OM ? {2{|di}} : {di[0],di[1]}) : di;

	assign CO = CI ^ DH[0] ^ DL[1];
	assign BO = DL[1] ? CI : !CI;
endmodule

module rvbmu_generic_slice #(
	parameter integer N = 8
) (
	input            RM = '0,  // Reverse-Mode
	input            GM = '0,  // Generate-Mode
	input            OM = '0,  // Or-combine-Mode

	input  [2*N-1:0] DI = '0,  // Data-In
	output [2*N-1:0] DO,       // Data-Out

	input            LI = '0,  // rotate-Left-In
	input            RI = '0,  // rotate-Right-In
	input    [N-1:0] BI = '0,  // control-Bits-In

	output           LO,       // rotate-Left-Out
	output           RO,       // rotate-Right-Out
	output   [N-1:0] BO        // control-Bits-Out
);
	wire [N:0] carry;
	assign carry[0] = 0;

	// control logic

	reg [2*N-1:0] datain;
	reg ctrl_ror, ctrl_rol, ctrl_neg;

	assign LO = ctrl_rol;
	assign RO = ctrl_ror;

	always @* begin
		ctrl_neg = 0;
		ctrl_ror = 0;
		ctrl_rol = 0;
		datain = DI;

		if (GM) begin
			if (^datain) begin
				ctrl_neg = RM;
			end else
			if (!RM) begin
				ctrl_neg = 1;
				ctrl_rol = 1;
				ctrl_ror = datain[2*N-1];
			end

			if (ctrl_neg)
				datain[N-1] = !datain[N-1];
		end
	end

	// actual slice

	rvbmu_generic_cell cells [N-1:0] (
		.OM(OM),

		.DH({datain[0], datain[2*N-1:1]}),
		.DL({datain[2*N-2:0], datain[2*N-1]}),
		.DO(DO),

		.CI(carry[N-1:0]),
		.LI(GM ? LO : LI),
		.RI(GM ? RO : RI),
		.BI(GM ? BO : BI),

		.CO(carry[N:1]),
		.BO(BO)
	);
endmodule

module rvbmu_generic_stage #(
	parameter integer N = 8,
	parameter integer M = 4
) (
	input              RM = '0,   // Reverse-Mode
	input              GM = '0,   // Generate-Mode
	input              OM = '0,   // Or-combine-Mode

	input  [2*N*M-1:0] DI = '0,   // Data-In
	output [2*N*M-1:0] DO,        // Data-Out

	input      [M-1:0] LI = '0,   // rotate-Left-In
	input      [M-1:0] RI = '0,   // rotate-Right-In
	input    [N*M-1:0] BI = '0,   // control-Bits-In

	output     [M-1:0] LO,        // rotate-Left-Out
	output     [M-1:0] RO,        // rotate-Right-Out
	output   [N*M-1:0] BO         // control-Bits-Out
);
	rvbmu_generic_slice #(N) slices [M-1:0] (
		.RM(RM),
		.GM(GM),
		.OM(OM),

		.DI(DI),
		.DO(DO),

		.LI(LI),
		.RI(RI),
		.BI(BI),

		.LO(LO),
		.RO(RO),
		.BO(BO)
	);
endmodule

module rvbmu_generic_gen32 (
	input          RM = '0,  // Reverse-Mode
	input   [31:0] DI = '0,  // Data-In

	output  [15:0] S0B,  // Stage-0-control-Bits
	output  [15:0] S1B,  // Stage-1-control-Bits
	output  [15:0] S2B,  // Stage-2-control-Bits
	output  [15:0] S3B,  // Stage-3-control-Bits
	output  [15:0] S4B,  // Stage-4-control-Bits

	output  [ 0:0] S0L,  // Stage-0-rotate-Left
	output  [ 1:0] S1L,  // Stage-1-rotate-Left
	output  [ 3:0] S2L,  // Stage-2-rotate-Left
	output  [ 7:0] S3L,  // Stage-3-rotate-Left
	output  [15:0] S4L,  // Stage-4-rotate-Left

	output  [ 0:0] S0R,  // Stage-0-rotate-Right
	output  [ 1:0] S1R,  // Stage-1-rotate-Right
	output  [ 3:0] S2R,  // Stage-2-rotate-Right
	output  [ 7:0] S3R,  // Stage-3-rotate-Right
	output  [15:0] S4R   // Stage-4-rotate-Right
);
	wire [31:0] D1, D2, D3, D4, DO;

	rvbmu_generic_stage #(16,  1) stage0 (.RM(RM), .GM(1'b1), .BI(16'b0), .LI( 1'b0), .RI( 1'b0), .DI(DI), .DO(D1),  .LO(S0L), .RO(S0R), .BO(S0B));
	rvbmu_generic_stage #( 8,  2) stage1 (.RM(RM), .GM(1'b1), .BI(16'b0), .LI( 2'b0), .RI( 2'b0), .DI(D1), .DO(D2),  .LO(S1L), .RO(S1R), .BO(S1B));
	rvbmu_generic_stage #( 4,  4) stage2 (.RM(RM), .GM(1'b1), .BI(16'b0), .LI( 4'b0), .RI( 4'b0), .DI(D2), .DO(D3),  .LO(S2L), .RO(S2R), .BO(S2B));
	rvbmu_generic_stage #( 2,  8) stage3 (.RM(RM), .GM(1'b1), .BI(16'b0), .LI( 8'b0), .RI( 8'b0), .DI(D3), .DO(D4),  .LO(S3L), .RO(S3R), .BO(S3B));
	rvbmu_generic_stage #( 1, 16) stage4 (.RM(RM), .GM(1'b1), .BI(16'b0), .LI(16'b0), .RI(16'b0), .DI(D4), .DO(DO),  .LO(S4L), .RO(S4R), .BO(S4B));
endmodule

module rvbmu_generic_exe32 (
	input   [31:0] DI = '0,   // Data-In
	output  [31:0] DO,        // Data-Out

	input   [15:0] S0B = '0,  // Stage-0-control-Bits
	input   [15:0] S1B = '0,  // Stage-1-control-Bits
	input   [15:0] S2B = '0,  // Stage-2-control-Bits
	input   [15:0] S3B = '0,  // Stage-3-control-Bits
	input   [15:0] S4B = '0,  // Stage-4-control-Bits

	input   [ 0:0] S0L = '0,  // Stage-0-rotate-Left
	input   [ 1:0] S1L = '0,  // Stage-1-rotate-Left
	input   [ 3:0] S2L = '0,  // Stage-2-rotate-Left
	input   [ 7:0] S3L = '0,  // Stage-3-rotate-Left
	input   [15:0] S4L = '0,  // Stage-4-rotate-Left

	input   [ 0:0] S0R = '0,  // Stage-0-rotate-Right
	input   [ 1:0] S1R = '0,  // Stage-1-rotate-Right
	input   [ 3:0] S2R = '0,  // Stage-2-rotate-Right
	input   [ 7:0] S3R = '0,  // Stage-3-rotate-Right
	input   [15:0] S4R = '0   // Stage-4-rotate-Right
);
	wire [31:0] D1, D2, D3, D4;

	rvbmu_generic_stage #(16,  1) stage0 (.BI(S0B), .LI(S0L), .RI(S0R), .DI(DI), .DO(D1));
	rvbmu_generic_stage #( 8,  2) stage1 (.BI(S1B), .LI(S1L), .RI(S1R), .DI(D1), .DO(D2));
	rvbmu_generic_stage #( 4,  4) stage2 (.BI(S2B), .LI(S2L), .RI(S2R), .DI(D2), .DO(D3));
	rvbmu_generic_stage #( 2,  8) stage3 (.BI(S3B), .LI(S3L), .RI(S3R), .DI(D3), .DO(D4));
	rvbmu_generic_stage #( 1, 16) stage4 (.BI(S4B), .LI(S4L), .RI(S4R), .DI(D4), .DO(DO));
endmodule
