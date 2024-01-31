
`timescale 1ns / 1ps

module SINCPDE_IMAC(clk, rst, z_en, x, y, mode, samples, z);

parameter NUM_SAMPLES = 11;
parameter BITS_SAMPLES = 4;

input clk, rst;
input [17:0] x;
input [47:0] y;
input [1:0] mode;

input [NUM_SAMPLES*18-1:0] samples;

output reg [47:0] z;
output reg z_en;

// -----------------------------------------------------------------------

reg signed [17:0] buf_x;
reg [47:0] buf_y;
reg [1:0] buf_mode;

wire [9:0] off = 
	buf_mode == 0 ?   0 :
	buf_mode == 1 ? 256 :
	buf_mode == 2 ? 512 : 'bx;

reg signed [17:0] buf_samples [NUM_SAMPLES-1:0];

// -----------------------------------------------------------------------

reg [17:0] tab [3*256-1:0];

localparam tab_data = {
    18'h00000, 18'h3ff00, 18'h3fe06, 18'h3fd1b, 18'h3fc49, 18'h3fb98, 18'h3fb11, 18'h3fab8, 18'h3fa92, 18'h3faa1, 18'h3fae6, 18'h3fb5f, 18'h3fc07, 18'h3fcda, 18'h3fdd0, 18'h3fee0,
    18'h00000, 18'h00125, 18'h00244, 18'h00352, 18'h00444, 18'h00510, 18'h005ae, 18'h00616, 18'h00644, 18'h00635, 18'h005e7, 18'h0055d, 18'h0049c, 18'h003a8, 18'h0028b, 18'h0014f,
    18'h00000, 18'h3feaa, 18'h3fd59, 18'h3fc1c, 18'h3fafe, 18'h3fa0c, 18'h3f94f, 18'h3f8d1, 18'h3f898, 18'h3f8a7, 18'h3f900, 18'h3f9a0, 18'h3fa84, 18'h3fba3, 18'h3fcf6, 18'h3fe6f,
    18'h00000, 18'h0019c, 18'h00332, 18'h004b4, 18'h00610, 18'h0073a, 18'h00823, 18'h008c2, 18'h0090d, 18'h00901, 18'h0089a, 18'h007db, 18'h006c7, 18'h00567, 18'h003c7, 18'h001f4,
    18'h00000, 18'h3fdfc, 18'h3fbfa, 18'h3fa11, 18'h3f852, 18'h3f6d1, 18'h3f59e, 18'h3f4c9, 18'h3f45c, 18'h3f461, 18'h3f4d9, 18'h3f5c6, 18'h3f723, 18'h3f8e7, 18'h3fb03, 18'h3fd68,
    18'h00000, 18'h002b4, 18'h0056c, 18'h0080c, 18'h00a79, 18'h00c9a, 18'h00e57, 18'h00f98, 18'h0104c, 18'h01064, 18'h00fd9, 18'h00ea6, 18'h00ccd, 18'h00a59, 18'h00756, 18'h003da,
    18'h00000, 18'h3fbe6, 18'h3f7b0, 18'h3f383, 18'h3ef8a, 18'h3ebed, 18'h3e8d6, 18'h3e66d, 18'h3e4d7, 18'h3e434, 18'h3e4a0, 18'h3e631, 18'h3e8f4, 18'h3ecf1, 18'h3f224, 18'h3f885,
    18'h00000, 18'h0087a, 18'h011d1, 18'h01bdc, 18'h02669, 18'h03146, 18'h03c3a, 18'h0470a, 18'h0517c, 18'h05b56, 18'h06461, 18'h06c68, 18'h0733d, 18'h078b9, 18'h07cbc, 18'h07f2d,
    18'h08000, 18'h07f2d, 18'h07cbc, 18'h078b9, 18'h0733d, 18'h06c68, 18'h06461, 18'h05b56, 18'h0517c, 18'h0470a, 18'h03c3a, 18'h03146, 18'h02669, 18'h01bdc, 18'h011d1, 18'h0087a,
    18'h00000, 18'h3f885, 18'h3f224, 18'h3ecf1, 18'h3e8f4, 18'h3e631, 18'h3e4a0, 18'h3e434, 18'h3e4d7, 18'h3e66d, 18'h3e8d6, 18'h3ebed, 18'h3ef8a, 18'h3f383, 18'h3f7b0, 18'h3fbe6,
    18'h00000, 18'h003da, 18'h00756, 18'h00a59, 18'h00ccd, 18'h00ea6, 18'h00fd9, 18'h01064, 18'h0104c, 18'h00f98, 18'h00e57, 18'h00c9a, 18'h00a79, 18'h0080c, 18'h0056c, 18'h002b4,
    18'h00000, 18'h3fd68, 18'h3fb03, 18'h3f8e7, 18'h3f723, 18'h3f5c6, 18'h3f4d9, 18'h3f461, 18'h3f45c, 18'h3f4c9, 18'h3f59e, 18'h3f6d1, 18'h3f852, 18'h3fa11, 18'h3fbfa, 18'h3fdfc,
    18'h00000, 18'h001f4, 18'h003c7, 18'h00567, 18'h006c7, 18'h007db, 18'h0089a, 18'h00901, 18'h0090d, 18'h008c2, 18'h00823, 18'h0073a, 18'h00610, 18'h004b4, 18'h00332, 18'h0019c,
    18'h00000, 18'h3fe6f, 18'h3fcf6, 18'h3fba3, 18'h3fa84, 18'h3f9a0, 18'h3f900, 18'h3f8a7, 18'h3f898, 18'h3f8d1, 18'h3f94f, 18'h3fa0c, 18'h3fafe, 18'h3fc1c, 18'h3fd59, 18'h3feaa,
    18'h00000, 18'h0014f, 18'h0028b, 18'h003a8, 18'h0049c, 18'h0055d, 18'h005e7, 18'h00635, 18'h00644, 18'h00616, 18'h005ae, 18'h00510, 18'h00444, 18'h00352, 18'h00244, 18'h00125,
    18'h00000, 18'h3fee0, 18'h3fdd0, 18'h3fcda, 18'h3fc07, 18'h3fb5f, 18'h3fae6, 18'h3faa1, 18'h3fa92, 18'h3fab8, 18'h3fb11, 18'h3fb98, 18'h3fc49, 18'h3fd1b, 18'h3fe06, 18'h3ff00,
    18'h3f000, 18'h3f00f, 18'h3f0bc, 18'h3f202, 18'h3f3d8, 18'h3f62e, 18'h3f8ee, 18'h3fc00, 18'h3ff47, 18'h002a2, 18'h005f3, 18'h00917, 18'h00bef, 18'h00e5e, 18'h0104a, 18'h0119d,
    18'h01249, 18'h01242, 18'h01187, 18'h0101c, 18'h00e0a, 18'h00b64, 18'h00840, 18'h004bb, 18'h000f6, 18'h3fd16, 18'h3f93f, 18'h3f596, 18'h3f242, 18'h3ef65, 18'h3ed1c, 18'h3eb83,
    18'h3eaab, 18'h3eaa2, 18'h3eb6c, 18'h3ed06, 18'h3ef64, 18'h3f274, 18'h3f61b, 18'h3fa39, 18'h3fea8, 18'h0033d, 18'h007cf, 18'h00c2f, 18'h01031, 18'h013ac, 18'h0167b, 18'h0187c,
    18'h01999, 18'h019c0, 18'h018e9, 18'h01717, 18'h01454, 18'h010b6, 18'h00c59, 18'h00764, 18'h00203, 18'h3fc67, 18'h3f6c6, 18'h3f155, 18'h3ec4d, 18'h3e7e1, 18'h3e440, 18'h3e195,
    18'h3e000, 18'h3df9b, 18'h3e072, 18'h3e287, 18'h3e5d1, 18'h3ea3a, 18'h3efa0, 18'h3f5d8, 18'h3fcad, 18'h003e1, 18'h00b35, 18'h01261, 18'h0191f, 18'h01f29, 18'h0243e, 18'h02825,
    18'h02aaa, 18'h02ba8, 18'h02b04, 18'h028b3, 18'h024b8, 18'h01f26, 18'h0181f, 18'h00fd4, 18'h00684, 18'h3fc7c, 18'h3f20d, 18'h3e796, 18'h3dd77, 18'h3d414, 18'h3cbce, 18'h3c501,
    18'h3c000, 18'h3bd17, 18'h3bc7f, 18'h3be65, 18'h3c2e0, 18'h3c9f7, 18'h3d39a, 18'h3dfa7, 18'h3ede5, 18'h3fe09, 18'h00fb6, 18'h02284, 18'h035f8, 18'h04992, 18'h05ccc, 18'h06f1d,
    18'h08000, 18'h08ef4, 18'h09b83, 18'h0a546, 18'h0abe5, 18'h0af1c, 18'h0aebc, 18'h0aab0, 18'h0a2f9, 18'h097b2, 18'h0890e, 18'h07757, 18'h062ec, 18'h04c40, 18'h033d4, 18'h01a37,
    18'h00000, 18'h3e5c9, 18'h3cc2c, 18'h3b3c0, 18'h39d14, 18'h388a9, 18'h376f2, 18'h3684e, 18'h35d07, 18'h35550, 18'h35144, 18'h350e4, 18'h3541b, 18'h35aba, 18'h3647d, 18'h3710c,
    18'h38000, 18'h390e3, 18'h3a334, 18'h3b66e, 18'h3ca08, 18'h3dd7c, 18'h3f04a, 18'h001f7, 18'h0121b, 18'h02059, 18'h02c66, 18'h03609, 18'h03d20, 18'h0419b, 18'h04381, 18'h042e9,
    18'h04000, 18'h03aff, 18'h03432, 18'h02bec, 18'h02289, 18'h0186a, 18'h00df3, 18'h00384, 18'h3f97c, 18'h3f02c, 18'h3e7e1, 18'h3e0da, 18'h3db48, 18'h3d74d, 18'h3d4fc, 18'h3d458,
    18'h3d556, 18'h3d7db, 18'h3dbc2, 18'h3e0d7, 18'h3e6e1, 18'h3ed9f, 18'h3f4cb, 18'h3fc1f, 18'h00353, 18'h00a28, 18'h01060, 18'h015c6, 18'h01a2f, 18'h01d79, 18'h01f8e, 18'h02065,
    18'h02000, 18'h01e6b, 18'h01bc0, 18'h0181f, 18'h013b3, 18'h00eab, 18'h0093a, 18'h00399, 18'h3fdfd, 18'h3f89c, 18'h3f3a7, 18'h3ef4a, 18'h3ebac, 18'h3e8e9, 18'h3e717, 18'h3e640,
    18'h3e667, 18'h3e784, 18'h3e985, 18'h3ec54, 18'h3efcf, 18'h3f3d1, 18'h3f831, 18'h3fcc3, 18'h00158, 18'h005c7, 18'h009e5, 18'h00d8c, 18'h0109c, 18'h012fa, 18'h01494, 18'h0155e,
    18'h01555, 18'h0147d, 18'h012e4, 18'h0109b, 18'h00dbe, 18'h00a6a, 18'h006c1, 18'h002ea, 18'h3ff0a, 18'h3fb45, 18'h3f7c0, 18'h3f49c, 18'h3f1f6, 18'h3efe4, 18'h3ee79, 18'h3edbe,
    18'h3edb7, 18'h3ee63, 18'h3efb6, 18'h3f1a2, 18'h3f411, 18'h3f6e9, 18'h3fa0d, 18'h3fd5e, 18'h000b9, 18'h00400, 18'h00712, 18'h009d2, 18'h00c28, 18'h00dfe, 18'h00f44, 18'h00ff1,
    18'h3fc00, 18'h005dd, 18'h00fa9, 18'h01903, 18'h0218d, 18'h028ef, 18'h02ede, 18'h03317, 18'h0356c, 18'h035bc, 18'h033fc, 18'h03035, 18'h02a83, 18'h02314, 18'h01a2b, 18'h01018,
    18'h00539, 18'h3f9f5, 18'h3eeb8, 18'h3e3f0, 18'h3da0a, 18'h3d169, 18'h3ca6a, 18'h3c559, 18'h3c26f, 18'h3c1d5, 18'h3c39b, 18'h3c7bc, 18'h3ce1c, 18'h3d687, 18'h3e0b6, 18'h3ec4d,
    18'h3f8e4, 18'h00603, 18'h0132f, 18'h01fe7, 18'h02bac, 18'h03605, 18'h03e87, 18'h044d3, 18'h0489f, 18'h049b9, 18'h04806, 18'h04386, 18'h03c54, 18'h032a6, 18'h026cc, 18'h0192b,
    18'h00a3d, 18'h3fa8b, 18'h3eaa8, 18'h3db2d, 18'h3ccb3, 18'h3bfce, 18'h3b504, 18'h3accd, 18'h3a789, 18'h3a581, 18'h3a6dd, 18'h3abab, 18'h3b3d3, 18'h3bf21, 18'h3cd3e, 18'h3ddb7,
    18'h3f000, 18'h00378, 18'h0176c, 18'h02b23, 18'h03ddc, 18'h04edc, 18'h05d73, 18'h06901, 18'h070fd, 18'h074fe, 18'h074b8, 18'h07008, 18'h066f3, 18'h059a4, 18'h04870, 18'h033d5,
    18'h01c71, 18'h00304, 18'h3e867, 18'h3cd83, 18'h3b350, 18'h39ac6, 18'h384da, 18'h37272, 18'h3645e, 18'h35b50, 18'h357d4, 18'h35a4d, 18'h362ee, 18'h371b7, 18'h38675, 18'h3a0c1,
    18'h3c000, 18'h3e36c, 18'h00a11, 18'h032dc, 18'h05c9f, 18'h08617, 18'h0adfa, 18'h0d301, 18'h0f3f0, 18'h10fa0, 18'h1250c, 18'h13357, 18'h139d3, 18'h1380b, 18'h12dc2, 18'h11afe,
    18'h10000, 18'h0dd49, 18'h0b398, 18'h083df, 18'h04f44, 18'h01714, 18'h3dcbd, 18'h3a1c0, 18'h367a7, 18'h32ffe, 18'h2fc43, 18'h2cdd9, 18'h2a603, 18'h285d6, 18'h26e34, 18'h25fc2,
    18'h25ae6, 18'h25fc2, 18'h26e34, 18'h285d6, 18'h2a603, 18'h2cdd9, 18'h2fc43, 18'h32ffe, 18'h367a7, 18'h3a1c0, 18'h3dcbd, 18'h01714, 18'h04f44, 18'h083df, 18'h0b398, 18'h0dd49,
    18'h10000, 18'h11afe, 18'h12dc2, 18'h1380b, 18'h139d3, 18'h13357, 18'h1250c, 18'h10fa0, 18'h0f3f0, 18'h0d301, 18'h0adfa, 18'h08617, 18'h05c9f, 18'h032dc, 18'h00a11, 18'h3e36c,
    18'h3c000, 18'h3a0c1, 18'h38675, 18'h371b7, 18'h362ee, 18'h35a4d, 18'h357d4, 18'h35b50, 18'h3645e, 18'h37272, 18'h384da, 18'h39ac6, 18'h3b350, 18'h3cd83, 18'h3e867, 18'h00304,
    18'h01c71, 18'h033d5, 18'h04870, 18'h059a4, 18'h066f3, 18'h07008, 18'h074b8, 18'h074fe, 18'h070fd, 18'h06901, 18'h05d73, 18'h04edc, 18'h03ddc, 18'h02b23, 18'h0176c, 18'h00378,
    18'h3f000, 18'h3ddb7, 18'h3cd3e, 18'h3bf21, 18'h3b3d3, 18'h3abab, 18'h3a6dd, 18'h3a581, 18'h3a789, 18'h3accd, 18'h3b504, 18'h3bfce, 18'h3ccb3, 18'h3db2d, 18'h3eaa8, 18'h3fa8b,
    18'h00a3d, 18'h0192b, 18'h026cc, 18'h032a6, 18'h03c54, 18'h04386, 18'h04806, 18'h049b9, 18'h0489f, 18'h044d3, 18'h03e87, 18'h03605, 18'h02bac, 18'h01fe7, 18'h0132f, 18'h00603,
    18'h3f8e4, 18'h3ec4d, 18'h3e0b6, 18'h3d687, 18'h3ce1c, 18'h3c7bc, 18'h3c39b, 18'h3c1d5, 18'h3c26f, 18'h3c559, 18'h3ca6a, 18'h3d169, 18'h3da0a, 18'h3e3f0, 18'h3eeb8, 18'h3f9f5,
    18'h00539, 18'h01018, 18'h01a2b, 18'h02314, 18'h02a83, 18'h03035, 18'h033fc, 18'h035bc, 18'h0356c, 18'h03317, 18'h02ede, 18'h028ef, 18'h0218d, 18'h01903, 18'h00fa9, 18'h005dd
};

integer i1;

initial begin
	for (i1 = 0; i1 < 3*256; i1 = i1 + 1)
		tab[i1] <= tab_data >> (18*3*256 - (i1+1)*18);
	// #1;
	// for (i1 = 0; i1 < 3*256; i1 = i1 + 1)
	// 	$display("%d %d", i1, $signed(tab[i1]));
end

// -----------------------------------------------------------------------

// DSP1: 18*16 multiplier with 6 bit right shift

reg signed [17:0] dsp1_in1;
reg signed [15:0] dsp1_in2;
wire signed [24:0] dsp1_out;

/********************************* BEGIN TEMPLATE: xilinx_dsp48e1.v *********************************/

// Cascade: 30-bit (each) output: Cascade Ports
wire [29:0] DSP1_ACOUT;                               // 30-bit output: A port cascade output
wire [17:0] DSP1_BCOUT;                               // 18-bit output: B port cascade output
wire        DSP1_CARRYCASCOUT;                        // 1-bit output: Cascade carry output
wire        DSP1_MULTSIGNOUT;                         // 1-bit output: Multiplier sign cascade output
wire [47:0] DSP1_PCOUT;                               // 48-bit output: Cascade output

// Control: 1-bit (each) output: Control Inputs/Status Bits
wire        DSP1_OVERFLOW;                            // 1-bit output: Overflow in add/acc output
wire        DSP1_PATTERNBDETECT;                      // 1-bit output: Pattern bar detect output
wire        DSP1_PATTERNDETECT;                       // 1-bit output: Pattern detect output
wire        DSP1_UNDERFLOW;                           // 1-bit output: Underflow in add/acc output

// Data: 4-bit (each) output: Data Ports
wire  [3:0] DSP1_CARRYOUT;                            // 4-bit output: Carry output
wire [47:0] DSP1_P;                                   // 48-bit output: Primary data output
assign dsp1_out = $signed(DSP1_P) >>> 6;

// Cascade: 30-bit (each) input: Cascade Ports
wire [29:0] DSP1_ACIN            = 0;                 // 30-bit input: A cascade data input
wire [17:0] DSP1_BCIN            = 0;                 // 18-bit input: B cascade input
wire        DSP1_CARRYCASCIN     = 0;                 // 1-bit input: Cascade carry input
wire        DSP1_MULTSIGNIN      = 0;                 // 1-bit input: Multiplier sign input
wire [47:0] DSP1_PCIN            = 0;                 // 48-bit input: P cascade input

// Control: 4-bit (each) input: Control Inputs/Status Bits
wire  [3:0] DSP1_ALUMODE         = 4'b     0000;      // 4-bit input: ALU control input
wire  [2:0] DSP1_CARRYINSEL      = 3'b      000;      // 3-bit input: Carry select input
wire        DSP1_CEINMODE        = 1;                 // 1-bit input: Clock enable input for INMODEREG
wire        DSP1_CLK             = clk;               // 1-bit input: Clock input
wire  [4:0] DSP1_INMODE          = 5'b   0_0000;      // 5-bit input: INMODE control input
wire  [6:0] DSP1_OPMODE          = 7'b 000_0101;      // 7-bit input: Operation mode input
wire        DSP1_RSTINMODE       = 0;                 // 1-bit input: Reset input for INMODEREG

// Data: 30-bit (each) input: Data Ports
wire [29:0] DSP1_A               = $signed(dsp1_in1); // 30-bit input: A data input
wire [17:0] DSP1_B               = $signed(dsp1_in2); // 18-bit input: B data input
wire [47:0] DSP1_C               = 0;                 // 48-bit input: C data input
wire        DSP1_CARRYIN         = 0;                 // 1-bit input: Carry input signal
wire [24:0] DSP1_D               = 0;                 // 25-bit input: D data input

// Reset/Clock Enable: 1-bit (each) input: Reset/Clock Enable Inputs
wire        DSP1_CEA1            = 1;                 // 1-bit input: Clock enable input for 1st stage AREG
wire        DSP1_CEA2            = 1;                 // 1-bit input: Clock enable input for 2nd stage AREG
wire        DSP1_CEAD            = 1;                 // 1-bit input: Clock enable input for ADREG
wire        DSP1_CEALUMODE       = 1;                 // 1-bit input: Clock enable input for ALUMODERE
wire        DSP1_CEB1            = 1;                 // 1-bit input: Clock enable input for 1st stage BREG
wire        DSP1_CEB2            = 1;                 // 1-bit input: Clock enable input for 2nd stage BREG
wire        DSP1_CEC             = 1;                 // 1-bit input: Clock enable input for CREG
wire        DSP1_CECARRYIN       = 1;                 // 1-bit input: Clock enable input for CARRYINREG
wire        DSP1_CECTRL          = 1;                 // 1-bit input: Clock enable input for OPMODEREG and CARRYINSELREG
wire        DSP1_CED             = 1;                 // 1-bit input: Clock enable input for DREG
wire        DSP1_CEM             = 1;                 // 1-bit input: Clock enable input for MREG
wire        DSP1_CEP             = 1;                 // 1-bit input: Clock enable input for PREG
wire        DSP1_RSTA            = 0;                 // 1-bit input: Reset input for AREG
wire        DSP1_RSTALLCARRYIN   = 0;                 // 1-bit input: Reset input for CARRYINREG
wire        DSP1_RSTALUMODE      = 0;                 // 1-bit input: Reset input for ALUMODEREG
wire        DSP1_RSTB            = 0;                 // 1-bit input: Reset input for BREG
wire        DSP1_RSTC            = 0;                 // 1-bit input: Reset input for CREG
wire        DSP1_RSTCTRL         = 0;                 // 1-bit input: Reset input for OPMODEREG and CARRYINSELREG
wire        DSP1_RSTD            = 0;                 // 1-bit input: Reset input for DREG and ADREG
wire        DSP1_RSTM            = 0;                 // 1-bit input: Reset input for MREG
wire        DSP1_RSTP            = 0;                 // 1-bit input: Reset input for PREG

DSP48E1 #(
	// Feature Control Attributes: Data Path Selection
	.A_INPUT("DIRECT"),               // Selects A input source, "DIRECT" (A port) or "CASCADE" (ACIN port)
	.B_INPUT("DIRECT"),               // Selects B input source, "DIRECT" (B port) or "CASCADE" (BCIN port)
	.USE_DPORT("FALSE"),              // Select D port usage (TRUE or FALSE)
	.USE_MULT("MULTIPLY"),            // Select multiplier usage ("MULTIPLY", "DYNAMIC", or "NONE")
	// Pattern Detector Attributes: Pattern Detection Configuration
	.AUTORESET_PATDET("NO_RESET"),    // "NO_RESET", "RESET_MATCH", "RESET_NOT_MATCH" 
	.MASK(48'h3fffffffffff),          // 48-bit mask value for pattern detect (1=ignore)
	.PATTERN(48'h000000000000),       // 48-bit pattern match for pattern detect
	.SEL_MASK("MASK"),                // "C", "MASK", "ROUNDING_MODE1", "ROUNDING_MODE2" 
	.SEL_PATTERN("PATTERN"),          // Select pattern value ("PATTERN" or "C")
	.USE_PATTERN_DETECT("NO_PATDET"), // Enable pattern detect ("PATDET" or "NO_PATDET")
	// Register Control Attributes: Pipeline Register Configuration
	.ACASCREG(1),                     // Number of pipeline stages between A/ACIN and ACOUT (0, 1 or 2)
	.ADREG(1),                        // Number of pipeline stages for pre-adder (0 or 1)
	.ALUMODEREG(1),                   // Number of pipeline stages for ALUMODE (0 or 1)
	.AREG(1),                         // Number of pipeline stages for A (0, 1 or 2)
	.BCASCREG(1),                     // Number of pipeline stages between B/BCIN and BCOUT (0, 1 or 2)
	.BREG(1),                         // Number of pipeline stages for B (0, 1 or 2)
	.CARRYINREG(1),                   // Number of pipeline stages for CARRYIN (0 or 1)
	.CARRYINSELREG(1),                // Number of pipeline stages for CARRYINSEL (0 or 1)
	.CREG(1),                         // Number of pipeline stages for C (0 or 1)
	.DREG(1),                         // Number of pipeline stages for D (0 or 1)
	.INMODEREG(1),                    // Number of pipeline stages for INMODE (0 or 1)
	.MREG(1),                         // Number of multiplier pipeline stages (0 or 1)
	.OPMODEREG(1),                    // Number of pipeline stages for OPMODE (0 or 1)
	.PREG(1),                         // Number of pipeline stages for P (0 or 1)
	.USE_SIMD("ONE48")                // SIMD selection ("ONE48", "TWO24", "FOUR12")
) DSP1_DSP48E1 (
	// Cascade: 30-bit (each) output: Cascade Ports
	.ACOUT(DSP1_ACOUT),                   // 30-bit output: A port cascade output
	.BCOUT(DSP1_BCOUT),                   // 18-bit output: B port cascade output
	.CARRYCASCOUT(DSP1_CARRYCASCOUT),     // 1-bit output: Cascade carry output
	.MULTSIGNOUT(DSP1_MULTSIGNOUT),       // 1-bit output: Multiplier sign cascade output
	.PCOUT(DSP1_PCOUT),                   // 48-bit output: Cascade output
	// Control: 1-bit (each) output: Control Inputs/Status Bits
	.OVERFLOW(DSP1_OVERFLOW),             // 1-bit output: Overflow in add/acc output
	.PATTERNBDETECT(DSP1_PATTERNBDETECT), // 1-bit output: Pattern bar detect output
	.PATTERNDETECT(DSP1_PATTERNDETECT),   // 1-bit output: Pattern detect output
	.UNDERFLOW(DSP1_UNDERFLOW),           // 1-bit output: Underflow in add/acc output
	// Data: 4-bit (each) output: Data Ports
	.CARRYOUT(DSP1_CARRYOUT),             // 4-bit output: Carry output
	.P(DSP1_P),                           // 48-bit output: Primary data output
	// Cascade: 30-bit (each) input: Cascade Ports
	.ACIN(DSP1_ACIN),                     // 30-bit input: A cascade data input
	.BCIN(DSP1_BCIN),                     // 18-bit input: B cascade input
	.CARRYCASCIN(DSP1_CARRYCASCIN),       // 1-bit input: Cascade carry input
	.MULTSIGNIN(DSP1_MULTSIGNIN),         // 1-bit input: Multiplier sign input
	.PCIN(DSP1_PCIN),                     // 48-bit input: P cascade input
	// Control: 4-bit (each) input: Control Inputs/Status Bits
	.ALUMODE(DSP1_ALUMODE),               // 4-bit input: ALU control input
	.CARRYINSEL(DSP1_CARRYINSEL),         // 3-bit input: Carry select input
	.CEINMODE(DSP1_CEINMODE),             // 1-bit input: Clock enable input for INMODEREG
	.CLK(DSP1_CLK),                       // 1-bit input: Clock input
	.INMODE(DSP1_INMODE),                 // 5-bit input: INMODE control input
	.OPMODE(DSP1_OPMODE),                 // 7-bit input: Operation mode input
	.RSTINMODE(DSP1_RSTINMODE),           // 1-bit input: Reset input for INMODEREG
	// Data: 30-bit (each) input: Data Ports
	.A(DSP1_A),                           // 30-bit input: A data input
	.B(DSP1_B),                           // 18-bit input: B data input
	.C(DSP1_C),                           // 48-bit input: C data input
	.CARRYIN(DSP1_CARRYIN),               // 1-bit input: Carry input signal
	.D(DSP1_D),                           // 25-bit input: D data input
	// Reset/Clock Enable: 1-bit (each) input: Reset/Clock Enable Inputs
	.CEA1(DSP1_CEA1),                     // 1-bit input: Clock enable input for 1st stage AREG
	.CEA2(DSP1_CEA2),                     // 1-bit input: Clock enable input for 2nd stage AREG
	.CEAD(DSP1_CEAD),                     // 1-bit input: Clock enable input for ADREG
	.CEALUMODE(DSP1_CEALUMODE),           // 1-bit input: Clock enable input for ALUMODERE
	.CEB1(DSP1_CEB1),                     // 1-bit input: Clock enable input for 1st stage BREG
	.CEB2(DSP1_CEB2),                     // 1-bit input: Clock enable input for 2nd stage BREG
	.CEC(DSP1_CEC),                       // 1-bit input: Clock enable input for CREG
	.CECARRYIN(DSP1_CECARRYIN),           // 1-bit input: Clock enable input for CARRYINREG
	.CECTRL(DSP1_CECTRL),                 // 1-bit input: Clock enable input for OPMODEREG and CARRYINSELREG
	.CED(DSP1_CED),                       // 1-bit input: Clock enable input for DREG
	.CEM(DSP1_CEM),                       // 1-bit input: Clock enable input for MREG
	.CEP(DSP1_CEP),                       // 1-bit input: Clock enable input for PREG
	.RSTA(DSP1_RSTA),                     // 1-bit input: Reset input for AREG
	.RSTALLCARRYIN(DSP1_RSTALLCARRYIN),   // 1-bit input: Reset input for CARRYINREG
	.RSTALUMODE(DSP1_RSTALUMODE),         // 1-bit input: Reset input for ALUMODEREG
	.RSTB(DSP1_RSTB),                     // 1-bit input: Reset input for BREG
	.RSTC(DSP1_RSTC),                     // 1-bit input: Reset input for CREG
	.RSTCTRL(DSP1_RSTCTRL),               // 1-bit input: Reset input for OPMODEREG and CARRYINSELREG
	.RSTD(DSP1_RSTD),                     // 1-bit input: Reset input for DREG and ADREG
	.RSTM(DSP1_RSTM),                     // 1-bit input: Reset input for MREG
	.RSTP(DSP1_RSTP)                      // 1-bit input: Reset input for PREG
);

/********************************* END TEMPLATE: xilinx_dsp48e1.v *********************************/

// DSP2: 25*16 macc

wire signed [24:0] dsp2_in1;
wire signed [17:0] dsp2_in2;
wire signed [47:0] dsp2_in3;
wire signed [47:0] dsp2_out;
wire dsp2_load;

/********************************* BEGIN TEMPLATE: xilinx_dsp48e1.v *********************************/

// Cascade: 30-bit (each) output: Cascade Ports
wire [29:0] DSP2_ACOUT;                               // 30-bit output: A port cascade output
wire [17:0] DSP2_BCOUT;                               // 18-bit output: B port cascade output
wire        DSP2_CARRYCASCOUT;                        // 1-bit output: Cascade carry output
wire        DSP2_MULTSIGNOUT;                         // 1-bit output: Multiplier sign cascade output
wire [47:0] DSP2_PCOUT;                               // 48-bit output: Cascade output

// Control: 1-bit (each) output: Control Inputs/Status Bits
wire        DSP2_OVERFLOW;                            // 1-bit output: Overflow in add/acc output
wire        DSP2_PATTERNBDETECT;                      // 1-bit output: Pattern bar detect output
wire        DSP2_PATTERNDETECT;                       // 1-bit output: Pattern detect output
wire        DSP2_UNDERFLOW;                           // 1-bit output: Underflow in add/acc output

// Data: 4-bit (each) output: Data Ports
wire  [3:0] DSP2_CARRYOUT;                            // 4-bit output: Carry output
wire [47:0] DSP2_P;                                   // 48-bit output: Primary data output
assign dsp2_out = $signed(DSP2_P);

// Cascade: 30-bit (each) input: Cascade Ports
wire [29:0] DSP2_ACIN            = 0;                 // 30-bit input: A cascade data input
wire [17:0] DSP2_BCIN            = 0;                 // 18-bit input: B cascade input
wire        DSP2_CARRYCASCIN     = 0;                 // 1-bit input: Cascade carry input
wire        DSP2_MULTSIGNIN      = 0;                 // 1-bit input: Multiplier sign input
wire [47:0] DSP2_PCIN            = 0;                 // 48-bit input: P cascade input

// Control: 4-bit (each) input: Control Inputs/Status Bits
wire  [3:0] DSP2_ALUMODE         = 4'b     0000;      // 4-bit input: ALU control input
wire  [2:0] DSP2_CARRYINSEL      = 3'b      000;      // 3-bit input: Carry select input
wire        DSP2_CEINMODE        = 1;                 // 1-bit input: Clock enable input for INMODEREG
wire        DSP2_CLK             = clk;               // 1-bit input: Clock input
wire  [4:0] DSP2_INMODE          = 5'b   0_0000;      // 5-bit input: INMODE control input
wire  [6:0] DSP2_OPMODE          = dsp2_load === 1 ? 7'b011_0101 : 7'b010_0101; // 7-bit input: Operation mode input
wire        DSP2_RSTINMODE       = 0;                 // 1-bit input: Reset input for INMODEREG

// Data: 30-bit (each) input: Data Ports
wire [29:0] DSP2_A               = $signed(dsp2_in1); // 30-bit input: A data input
wire [17:0] DSP2_B               = $signed(dsp2_in2); // 18-bit input: B data input
wire [47:0] DSP2_C               = $signed(dsp2_in3); // 48-bit input: C data input
wire        DSP2_CARRYIN         = 0;                 // 1-bit input: Carry input signal
wire [24:0] DSP2_D               = 0;                 // 25-bit input: D data input

// Reset/Clock Enable: 1-bit (each) input: Reset/Clock Enable Inputs
wire        DSP2_CEA1            = 1;                 // 1-bit input: Clock enable input for 1st stage AREG
wire        DSP2_CEA2            = 1;                 // 1-bit input: Clock enable input for 2nd stage AREG
wire        DSP2_CEAD            = 1;                 // 1-bit input: Clock enable input for ADREG
wire        DSP2_CEALUMODE       = 1;                 // 1-bit input: Clock enable input for ALUMODERE
wire        DSP2_CEB1            = 1;                 // 1-bit input: Clock enable input for 1st stage BREG
wire        DSP2_CEB2            = 1;                 // 1-bit input: Clock enable input for 2nd stage BREG
wire        DSP2_CEC             = 1;                 // 1-bit input: Clock enable input for CREG
wire        DSP2_CECARRYIN       = 1;                 // 1-bit input: Clock enable input for CARRYINREG
wire        DSP2_CECTRL          = 1;                 // 1-bit input: Clock enable input for OPMODEREG and CARRYINSELREG
wire        DSP2_CED             = 1;                 // 1-bit input: Clock enable input for DREG
wire        DSP2_CEM             = 1;                 // 1-bit input: Clock enable input for MREG
wire        DSP2_CEP             = 1;                 // 1-bit input: Clock enable input for PREG
wire        DSP2_RSTA            = 0;                 // 1-bit input: Reset input for AREG
wire        DSP2_RSTALLCARRYIN   = 0;                 // 1-bit input: Reset input for CARRYINREG
wire        DSP2_RSTALUMODE      = 0;                 // 1-bit input: Reset input for ALUMODEREG
wire        DSP2_RSTB            = 0;                 // 1-bit input: Reset input for BREG
wire        DSP2_RSTC            = 0;                 // 1-bit input: Reset input for CREG
wire        DSP2_RSTCTRL         = 0;                 // 1-bit input: Reset input for OPMODEREG and CARRYINSELREG
wire        DSP2_RSTD            = 0;                 // 1-bit input: Reset input for DREG and ADREG
wire        DSP2_RSTM            = 0;                 // 1-bit input: Reset input for MREG
wire        DSP2_RSTP            = 0;                 // 1-bit input: Reset input for PREG

DSP48E1 #(
	// Feature Control Attributes: Data Path Selection
	.A_INPUT("DIRECT"),               // Selects A input source, "DIRECT" (A port) or "CASCADE" (ACIN port)
	.B_INPUT("DIRECT"),               // Selects B input source, "DIRECT" (B port) or "CASCADE" (BCIN port)
	.USE_DPORT("FALSE"),              // Select D port usage (TRUE or FALSE)
	.USE_MULT("MULTIPLY"),            // Select multiplier usage ("MULTIPLY", "DYNAMIC", or "NONE")
	// Pattern Detector Attributes: Pattern Detection Configuration
	.AUTORESET_PATDET("NO_RESET"),    // "NO_RESET", "RESET_MATCH", "RESET_NOT_MATCH" 
	.MASK(48'h3fffffffffff),          // 48-bit mask value for pattern detect (1=ignore)
	.PATTERN(48'h000000000000),       // 48-bit pattern match for pattern detect
	.SEL_MASK("MASK"),                // "C", "MASK", "ROUNDING_MODE1", "ROUNDING_MODE2" 
	.SEL_PATTERN("PATTERN"),          // Select pattern value ("PATTERN" or "C")
	.USE_PATTERN_DETECT("NO_PATDET"), // Enable pattern detect ("PATDET" or "NO_PATDET")
	// Register Control Attributes: Pipeline Register Configuration
	.ACASCREG(1),                     // Number of pipeline stages between A/ACIN and ACOUT (0, 1 or 2)
	.ADREG(1),                        // Number of pipeline stages for pre-adder (0 or 1)
	.ALUMODEREG(1),                   // Number of pipeline stages for ALUMODE (0 or 1)
	.AREG(1),                         // Number of pipeline stages for A (0, 1 or 2)
	.BCASCREG(1),                     // Number of pipeline stages between B/BCIN and BCOUT (0, 1 or 2)
	.BREG(1),                         // Number of pipeline stages for B (0, 1 or 2)
	.CARRYINREG(1),                   // Number of pipeline stages for CARRYIN (0 or 1)
	.CARRYINSELREG(1),                // Number of pipeline stages for CARRYINSEL (0 or 1)
	.CREG(1),                         // Number of pipeline stages for C (0 or 1)
	.DREG(1),                         // Number of pipeline stages for D (0 or 1)
	.INMODEREG(1),                    // Number of pipeline stages for INMODE (0 or 1)
	.MREG(1),                         // Number of multiplier pipeline stages (0 or 1)
	.OPMODEREG(1),                    // Number of pipeline stages for OPMODE (0 or 1)
	.PREG(1),                         // Number of pipeline stages for P (0 or 1)
	.USE_SIMD("ONE48")                // SIMD selection ("ONE48", "TWO24", "FOUR12")
) DSP2_DSP48E1 (
	// Cascade: 30-bit (each) output: Cascade Ports
	.ACOUT(DSP2_ACOUT),                   // 30-bit output: A port cascade output
	.BCOUT(DSP2_BCOUT),                   // 18-bit output: B port cascade output
	.CARRYCASCOUT(DSP2_CARRYCASCOUT),     // 1-bit output: Cascade carry output
	.MULTSIGNOUT(DSP2_MULTSIGNOUT),       // 1-bit output: Multiplier sign cascade output
	.PCOUT(DSP2_PCOUT),                   // 48-bit output: Cascade output
	// Control: 1-bit (each) output: Control Inputs/Status Bits
	.OVERFLOW(DSP2_OVERFLOW),             // 1-bit output: Overflow in add/acc output
	.PATTERNBDETECT(DSP2_PATTERNBDETECT), // 1-bit output: Pattern bar detect output
	.PATTERNDETECT(DSP2_PATTERNDETECT),   // 1-bit output: Pattern detect output
	.UNDERFLOW(DSP2_UNDERFLOW),           // 1-bit output: Underflow in add/acc output
	// Data: 4-bit (each) output: Data Ports
	.CARRYOUT(DSP2_CARRYOUT),             // 4-bit output: Carry output
	.P(DSP2_P),                           // 48-bit output: Primary data output
	// Cascade: 30-bit (each) input: Cascade Ports
	.ACIN(DSP2_ACIN),                     // 30-bit input: A cascade data input
	.BCIN(DSP2_BCIN),                     // 18-bit input: B cascade input
	.CARRYCASCIN(DSP2_CARRYCASCIN),       // 1-bit input: Cascade carry input
	.MULTSIGNIN(DSP2_MULTSIGNIN),         // 1-bit input: Multiplier sign input
	.PCIN(DSP2_PCIN),                     // 48-bit input: P cascade input
	// Control: 4-bit (each) input: Control Inputs/Status Bits
	.ALUMODE(DSP2_ALUMODE),               // 4-bit input: ALU control input
	.CARRYINSEL(DSP2_CARRYINSEL),         // 3-bit input: Carry select input
	.CEINMODE(DSP2_CEINMODE),             // 1-bit input: Clock enable input for INMODEREG
	.CLK(DSP2_CLK),                       // 1-bit input: Clock input
	.INMODE(DSP2_INMODE),                 // 5-bit input: INMODE control input
	.OPMODE(DSP2_OPMODE),                 // 7-bit input: Operation mode input
	.RSTINMODE(DSP2_RSTINMODE),           // 1-bit input: Reset input for INMODEREG
	// Data: 30-bit (each) input: Data Ports
	.A(DSP2_A),                           // 30-bit input: A data input
	.B(DSP2_B),                           // 18-bit input: B data input
	.C(DSP2_C),                           // 48-bit input: C data input
	.CARRYIN(DSP2_CARRYIN),               // 1-bit input: Carry input signal
	.D(DSP2_D),                           // 25-bit input: D data input
	// Reset/Clock Enable: 1-bit (each) input: Reset/Clock Enable Inputs
	.CEA1(DSP2_CEA1),                     // 1-bit input: Clock enable input for 1st stage AREG
	.CEA2(DSP2_CEA2),                     // 1-bit input: Clock enable input for 2nd stage AREG
	.CEAD(DSP2_CEAD),                     // 1-bit input: Clock enable input for ADREG
	.CEALUMODE(DSP2_CEALUMODE),           // 1-bit input: Clock enable input for ALUMODERE
	.CEB1(DSP2_CEB1),                     // 1-bit input: Clock enable input for 1st stage BREG
	.CEB2(DSP2_CEB2),                     // 1-bit input: Clock enable input for 2nd stage BREG
	.CEC(DSP2_CEC),                       // 1-bit input: Clock enable input for CREG
	.CECARRYIN(DSP2_CECARRYIN),           // 1-bit input: Clock enable input for CARRYINREG
	.CECTRL(DSP2_CECTRL),                 // 1-bit input: Clock enable input for OPMODEREG and CARRYINSELREG
	.CED(DSP2_CED),                       // 1-bit input: Clock enable input for DREG
	.CEM(DSP2_CEM),                       // 1-bit input: Clock enable input for MREG
	.CEP(DSP2_CEP),                       // 1-bit input: Clock enable input for PREG
	.RSTA(DSP2_RSTA),                     // 1-bit input: Reset input for AREG
	.RSTALLCARRYIN(DSP2_RSTALLCARRYIN),   // 1-bit input: Reset input for CARRYINREG
	.RSTALUMODE(DSP2_RSTALUMODE),         // 1-bit input: Reset input for ALUMODEREG
	.RSTB(DSP2_RSTB),                     // 1-bit input: Reset input for BREG
	.RSTC(DSP2_RSTC),                     // 1-bit input: Reset input for CREG
	.RSTCTRL(DSP2_RSTCTRL),               // 1-bit input: Reset input for OPMODEREG and CARRYINSELREG
	.RSTD(DSP2_RSTD),                     // 1-bit input: Reset input for DREG and ADREG
	.RSTM(DSP2_RSTM),                     // 1-bit input: Reset input for MREG
	.RSTP(DSP2_RSTP)                      // 1-bit input: Reset input for PREG
);

/********************************* END TEMPLATE: xilinx_dsp48e1.v *********************************/

// -----------------------------------------------------------------------

// pipeline inputs
reg [9:0] pipeline_idx;
reg signed [15:0] pipeline_subidx;
reg signed [17:0] pipeline_sample;
reg pipeline_clear;

// pipeline outputs
wire signed [47:0] pipeline_sum = dsp2_out;

// internal state
reg signed [17:0] pipeline_tabval;
reg signed [15:0] pipeline_subidx_q;
reg signed [17:0] pipeline_sample_q [4:0];
reg pipeline_clear_q [5:0];

assign dsp2_in1 = dsp1_out;
assign dsp2_in2 = pipeline_sample_q[4];
assign dsp2_in3 = buf_y;
assign dsp2_load = pipeline_clear_q[5];

integer i2;

always @(posedge clk) begin
	pipeline_tabval <= tab[pipeline_idx];
	pipeline_subidx_q <= pipeline_subidx;

	dsp1_in1 <= pipeline_tabval;
	dsp1_in2 <= pipeline_subidx_q;

	for (i2 = 0; i2 < 5; i2=i2+1)
		pipeline_sample_q[i2] <= i2 ? pipeline_sample_q[i2-1] : pipeline_sample;

	for (i2 = 0; i2 < 6; i2=i2+1)
		pipeline_clear_q[i2] <= i2 ? pipeline_clear_q[i2-1] : pipeline_clear;
end

// -----------------------------------------------------------------------

(* fsm_encoding="one-hot", safe_implementation="no" *)
integer state;

reg [BITS_SAMPLES-1:0] count;

integer i3;

always @(posedge clk)
begin
	pipeline_idx <= 10'bx;
	pipeline_subidx <= 16'bx;
	pipeline_sample <= 18'bx;
	pipeline_clear <= 1'bx;

	if (rst) begin
		buf_x <= x;
		buf_y <= y;
		buf_mode <= mode;
		for (i3 = 0; i3 < NUM_SAMPLES; i3 = i3+1)
			buf_samples[i3] <= samples >> (18*i3);
		state <= 100;
		z <= y;
		z_en <= 0;
	end else begin
		case (state)
			100: begin
				state <= 110;
				count <= 0;
			end
			110: begin
				state <= 120;
				pipeline_idx <= ($signed(buf_x >>> 10) + (count * 16) + (128 - 5*16)) | off;
				pipeline_subidx <= (1 << 10) - (buf_x & ((1 << 10) - 1));
				pipeline_sample <= buf_samples[count];
				pipeline_clear <= count == 0;
			end
			120: begin
				state <= count == NUM_SAMPLES-1 ? 130 : 110;
				count <= count == NUM_SAMPLES-1 ? 0 : count + 1;
				pipeline_idx <= ($signed(buf_x >>> 10) + (count * 16) + (128 - 5*16 + 1)) | off;
				pipeline_subidx <= buf_x & ((1 << 10) - 1);
				pipeline_sample <= buf_samples[count];
				pipeline_clear <= 0;
			end
			130: begin
				if (count == 8) begin
					z_en <= 1;
					z <= pipeline_sum;
					state <= 140;
				end
				count <= count + 1;
			end
			140: begin
				/* halt */
			end
		endcase
	end
end

endmodule

