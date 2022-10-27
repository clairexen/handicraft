module SAG4Fun32c (
	input ctrl_inv,
	input ctrl_msk,
	input [31:0] in_data,
	input [31:0] in_mask,
	output [31:0] out_data
);
	wire [31:0] r0_din, r0_dout, r1_dout, r2_dout, r3_dout, r4_dout;

	wire [15:0] m0_cin, m1_cin, m2_cin, m3_cin;
	wire [15:0] m0_cout, m1_cout, m2_cout, m3_cout;
	wire [15:0] m0_sout, m1_sout, m2_sout, m3_sout, m4_sout;
	wire [31:0] m0_dout, m1_dout, m2_dout, m3_dout;

	assign r0_din = in_data & (in_mask | {32{ctrl_inv | !ctrl_msk}});
	assign out_data = r4_dout & (in_mask | {32{!ctrl_inv | !ctrl_msk}});

	SAG4FunRow r0 (ctrl_inv, ctrl_inv ? m4_sout : m0_sout, 16'bx, r0_din,,,  r0_dout);
	SAG4FunRow r1 (ctrl_inv, ctrl_inv ? m3_sout : m1_sout, 16'bx, r0_dout,,, r1_dout);
	SAG4FunRow r2 (ctrl_inv, ctrl_inv ? m2_sout : m2_sout, 16'bx, r1_dout,,, r2_dout);
	SAG4FunRow r3 (ctrl_inv, ctrl_inv ? m1_sout : m3_sout, 16'bx, r2_dout,,, r3_dout);
	SAG4FunRow r4 (ctrl_inv, ctrl_inv ? m0_sout : m4_sout, 16'bx, r3_dout,,, r4_dout);

	assign m0_cin = 16'b 0000_0000_0000_0001 | (m0_cout << 1);
	assign m1_cin = 16'b 0000_0001_0000_0001 | (m1_cout << 1);
	assign m2_cin = 16'b 0001_0001_0001_0001 | (m2_cout << 1);
	assign m3_cin = 16'b 0101_0101_0101_0101 | (m3_cout << 1);

	SAG4FunRow m0 (1'b0, m0_sout, m0_cin, in_mask, m0_sout, m0_cout, m0_dout);
	SAG4FunRow m1 (1'b0, m1_sout, m1_cin, m0_dout, m1_sout, m1_cout, m1_dout);
	SAG4FunRow m2 (1'b0, m2_sout, m2_cin, m1_dout, m2_sout, m2_cout, m2_dout);
	SAG4FunRow m3 (1'b0, m3_sout, m3_cin, m2_dout, m3_sout, m3_cout, m3_dout);
	SAG4FunRow m4 (1'b0, m4_sout, 16'h FFFF, m3_dout, m4_sout,,);
endmodule

module SAG4FunRow #(
	parameter integer XLEN = 32
) (
	input  ctrl_unshuffle,

	input  [XLEN/2-1:0] in_swap,
	input  [XLEN/2-1:0] in_carry,
	input  [  XLEN-1:0] in_data,

	output [XLEN/2-1:0] out_swap,
	output [XLEN/2-1:0] out_carry,
	output [  XLEN-1:0] out_data

);
	wire [  XLEN-1:0] cells_in;
	wire [  XLEN-1:0] cells_out;

	function [XLEN-1:0] split;
		input [XLEN-1:0] in;
		if (XLEN == 32) split = {in[0], in[2], in[4], in[6], in[8],
			in[10], in[12], in[14], in[16], in[18], in[20], in[22], in[24],
			in[26], in[28], in[30], in[1], in[3], in[5], in[7], in[9],
			in[11], in[13], in[15], in[17], in[19], in[21], in[23], in[25],
			in[27], in[29], in[31]};
		else split = {in[0], in[2], in[4], in[6], in[8], in[10], in[12],
			in[14], in[16], in[18], in[20], in[22], in[24], in[26], in[28],
			in[30], in[32], in[34], in[36], in[38], in[40], in[42], in[44],
			in[46], in[48], in[50], in[52], in[54], in[56], in[58], in[60],
			in[62], in[1], in[3], in[5], in[7], in[9], in[11], in[13],
			in[15], in[17], in[19], in[21], in[23], in[25], in[27], in[29],
			in[31], in[33], in[35], in[37], in[39], in[41], in[43], in[45],
			in[47], in[49], in[51], in[53], in[55], in[57], in[59], in[61],
			in[63]};
	endfunction

	function [XLEN-1:0] merge;
		input [XLEN-1:0] in;
		if (XLEN == 32) merge = {in[0], in[16], in[1], in[17], in[2],
			in[18], in[3], in[19], in[4], in[20], in[5], in[21], in[6],
			in[22], in[7], in[23], in[8], in[24], in[9], in[25], in[10],
			in[26], in[11], in[27], in[12], in[28], in[13], in[29], in[14],
			in[30], in[15], in[31]};
		else merge = {in[0], in[32], in[1], in[33], in[2], in[34], in[3],
			in[35], in[4], in[36], in[5], in[37], in[6], in[38], in[7],
			in[39], in[8], in[40], in[9], in[41], in[10], in[42], in[11],
			in[43], in[12], in[44], in[13], in[45], in[14], in[46], in[15],
			in[47], in[16], in[48], in[17], in[49], in[18], in[50], in[19],
			in[51], in[20], in[52], in[21], in[53], in[22], in[54], in[23],
			in[55], in[24], in[56], in[25], in[57], in[26], in[58], in[27],
			in[59], in[28], in[60], in[29], in[61], in[30], in[62], in[31],
			in[63]};
	endfunction

	assign cells_in = ctrl_unshuffle ? merge(in_data) : in_data;
	assign out_data = ctrl_unshuffle ? cells_out : split(cells_out);

	SAG4FunCell cells [XLEN/2-1:0] (
		.in_swap   (in_swap  ),
		.in_carry  (in_carry ),
		.in_data   (cells_in ),
		.out_swap  (out_swap ),
		.out_carry (out_carry),
		.out_data  (cells_out)
	);
endmodule

module SAG4FunCell (
	input in_swap,
	input in_carry,
	input [1:0] in_data,

	output out_swap,
	output out_carry,
	input [1:0] out_data
);
	assign out_swap = in_carry ^ in_data[0];
	assign out_carry = out_swap ^ in_data[1];
	assign out_data = {in_data[!in_swap], in_data[in_swap]};
endmodule
