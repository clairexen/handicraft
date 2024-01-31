/*
 *  Copyright (C) 2017  Clifford Wolf <clifford@clifford.at>
 *
 *  Permission to use, copy, modify, and/or distribute this software for any
 *  purpose with or without fee is hereby granted, provided that the above
 *  copyright notice and this permission notice appear in all copies.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 *  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 *  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 *  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 *  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 *  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 */

module pdep_pext (
`ifdef PDEP_PEXT_PIPELINE
	input clk,
`endif
	input pdep, pext,
	input [63:0] din, mask,
	output [63:0] dout
);
	wire [31:0] s1, s2, s4, s8, s16, s32;

	pdep_pext_decoder decoder (
		.mask (mask),
		.s1   (s1  ),
		.s2   (s2  ),
		.s4   (s4  ),
		.s8   (s8  ),
		.s16  (s16 ),
		.s32  (s32 )
	);

`ifdef PDEP_PEXT_PIPELINE
	wire [63:0] bfly_do;
	reg [63:0] bfly_di;
	reg [63:0] mask_q;
	reg bfly_inv, pdep_q;
	reg [31:0] bfly_s1;
	reg [31:0] bfly_s2;
	reg [31:0] bfly_s4;
	reg [31:0] bfly_s8;
	reg [31:0] bfly_s16;
	reg [31:0] bfly_s32;

	assign dout = pdep_q ? bfly_do & mask_q : bfly_do;

	always @(posedge clk) begin
		bfly_di <=  pext ? din & mask : din;
		mask_q <= mask;
		pdep_q <= pdep;

`ifdef PDEP_PEXT_NO_GREV
		bfly_s1  <= s1;
		bfly_s2  <= s2;
		bfly_s4  <= s4;
		bfly_s8  <= s8;
		bfly_s16 <= s16;
		bfly_s32 <= s32;
`else
		bfly_s1  <= pdep || pext ? s1  : {32{~mask[0]}};
		bfly_s2  <= pdep || pext ? s2  : {32{~mask[1]}};
		bfly_s4  <= pdep || pext ? s4  : {32{~mask[2]}};
		bfly_s8  <= pdep || pext ? s8  : {32{~mask[3]}};
		bfly_s16 <= pdep || pext ? s16 : {32{~mask[4]}};
		bfly_s32 <= pdep || pext ? s32 : {32{~mask[5]}};
`endif
	end
`else
	wire [63:0] bfly_di, bfly_do;
	assign bfly_di =  pext ? din & mask : din;
	assign dout = pdep ? bfly_do & mask : bfly_do;

	wire bfly_inv = pext;

`ifdef PDEP_PEXT_NO_GREV
	wire [31:0] bfly_s1  = s1;
	wire [31:0] bfly_s2  = s2;
	wire [31:0] bfly_s4  = s4;
	wire [31:0] bfly_s8  = s8;
	wire [31:0] bfly_s16 = s16;
	wire [31:0] bfly_s32 = s32;
`else
	wire [31:0] bfly_s1  = pdep || pext ? s1  : {32{~mask[0]}};
	wire [31:0] bfly_s2  = pdep || pext ? s2  : {32{~mask[1]}};
	wire [31:0] bfly_s4  = pdep || pext ? s4  : {32{~mask[2]}};
	wire [31:0] bfly_s8  = pdep || pext ? s8  : {32{~mask[3]}};
	wire [31:0] bfly_s16 = pdep || pext ? s16 : {32{~mask[4]}};
	wire [31:0] bfly_s32 = pdep || pext ? s32 : {32{~mask[5]}};
`endif
`endif

	pdep_pext_butterfly butterfly (
		.inv     (bfly_inv),
		.din     (bfly_di ),
		.s1      (bfly_s1 ),
		.s2      (bfly_s2 ),
		.s4      (bfly_s4 ),
		.s8      (bfly_s8 ),
		.s16     (bfly_s16),
		.s32     (bfly_s32),
		.dout    (bfly_do )
	);
endmodule

// generated from genpps.py
module pdep_pext_ppc (
  input [63:0] din,
  output [511:0] dout
);
  function [15:0] carry_save_add;
    input [15:0] a, b;
    reg [7:0] x, y;
    begin
      x = a[15:8] ^ a[7:0] ^ b[7:0];
      y = ((a[15:8] & a[7:0]) | (a[15:8] & b[7:0]) | (a[7:0] & b[7:0])) << 1;
      carry_save_add[7:0] = x ^ y ^ b[15:8];
      carry_save_add[15:8] = ((x & y) | (x & b[15:8]) | (y & b[15:8])) << 1;
    end
  endfunction
  function [7:0] carry_save_get;
    input [15:0] a;
    begin
      carry_save_get = a[7:0] + a[15:8];
    end
  endfunction
  // inputs
  wire [15:0] e0s0 = {15'b0, din[0 +: 1]};
  wire [15:0] e1s0 = {15'b0, din[1 +: 1]};
  wire [15:0] e2s0 = {15'b0, din[2 +: 1]};
  wire [15:0] e3s0 = {15'b0, din[3 +: 1]};
  wire [15:0] e4s0 = {15'b0, din[4 +: 1]};
  wire [15:0] e5s0 = {15'b0, din[5 +: 1]};
  wire [15:0] e6s0 = {15'b0, din[6 +: 1]};
  wire [15:0] e7s0 = {15'b0, din[7 +: 1]};
  wire [15:0] e8s0 = {15'b0, din[8 +: 1]};
  wire [15:0] e9s0 = {15'b0, din[9 +: 1]};
  wire [15:0] e10s0 = {15'b0, din[10 +: 1]};
  wire [15:0] e11s0 = {15'b0, din[11 +: 1]};
  wire [15:0] e12s0 = {15'b0, din[12 +: 1]};
  wire [15:0] e13s0 = {15'b0, din[13 +: 1]};
  wire [15:0] e14s0 = {15'b0, din[14 +: 1]};
  wire [15:0] e15s0 = {15'b0, din[15 +: 1]};
  wire [15:0] e16s0 = {15'b0, din[16 +: 1]};
  wire [15:0] e17s0 = {15'b0, din[17 +: 1]};
  wire [15:0] e18s0 = {15'b0, din[18 +: 1]};
  wire [15:0] e19s0 = {15'b0, din[19 +: 1]};
  wire [15:0] e20s0 = {15'b0, din[20 +: 1]};
  wire [15:0] e21s0 = {15'b0, din[21 +: 1]};
  wire [15:0] e22s0 = {15'b0, din[22 +: 1]};
  wire [15:0] e23s0 = {15'b0, din[23 +: 1]};
  wire [15:0] e24s0 = {15'b0, din[24 +: 1]};
  wire [15:0] e25s0 = {15'b0, din[25 +: 1]};
  wire [15:0] e26s0 = {15'b0, din[26 +: 1]};
  wire [15:0] e27s0 = {15'b0, din[27 +: 1]};
  wire [15:0] e28s0 = {15'b0, din[28 +: 1]};
  wire [15:0] e29s0 = {15'b0, din[29 +: 1]};
  wire [15:0] e30s0 = {15'b0, din[30 +: 1]};
  wire [15:0] e31s0 = {15'b0, din[31 +: 1]};
  wire [15:0] e32s0 = {15'b0, din[32 +: 1]};
  wire [15:0] e33s0 = {15'b0, din[33 +: 1]};
  wire [15:0] e34s0 = {15'b0, din[34 +: 1]};
  wire [15:0] e35s0 = {15'b0, din[35 +: 1]};
  wire [15:0] e36s0 = {15'b0, din[36 +: 1]};
  wire [15:0] e37s0 = {15'b0, din[37 +: 1]};
  wire [15:0] e38s0 = {15'b0, din[38 +: 1]};
  wire [15:0] e39s0 = {15'b0, din[39 +: 1]};
  wire [15:0] e40s0 = {15'b0, din[40 +: 1]};
  wire [15:0] e41s0 = {15'b0, din[41 +: 1]};
  wire [15:0] e42s0 = {15'b0, din[42 +: 1]};
  wire [15:0] e43s0 = {15'b0, din[43 +: 1]};
  wire [15:0] e44s0 = {15'b0, din[44 +: 1]};
  wire [15:0] e45s0 = {15'b0, din[45 +: 1]};
  wire [15:0] e46s0 = {15'b0, din[46 +: 1]};
  wire [15:0] e47s0 = {15'b0, din[47 +: 1]};
  wire [15:0] e48s0 = {15'b0, din[48 +: 1]};
  wire [15:0] e49s0 = {15'b0, din[49 +: 1]};
  wire [15:0] e50s0 = {15'b0, din[50 +: 1]};
  wire [15:0] e51s0 = {15'b0, din[51 +: 1]};
  wire [15:0] e52s0 = {15'b0, din[52 +: 1]};
  wire [15:0] e53s0 = {15'b0, din[53 +: 1]};
  wire [15:0] e54s0 = {15'b0, din[54 +: 1]};
  wire [15:0] e55s0 = {15'b0, din[55 +: 1]};
  wire [15:0] e56s0 = {15'b0, din[56 +: 1]};
  wire [15:0] e57s0 = {15'b0, din[57 +: 1]};
  wire [15:0] e58s0 = {15'b0, din[58 +: 1]};
  wire [15:0] e59s0 = {15'b0, din[59 +: 1]};
  wire [15:0] e60s0 = {15'b0, din[60 +: 1]};
  wire [15:0] e61s0 = {15'b0, din[61 +: 1]};
  wire [15:0] e62s0 = {15'b0, din[62 +: 1]};
  wire [15:0] e63s0 = {15'b0, din[63 +: 1]};
  // forward pass
  wire [15:0] e1s1 = carry_save_add(e1s0, e0s0);
  wire [15:0] e3s1 = carry_save_add(e3s0, e2s0);
  wire [15:0] e5s1 = carry_save_add(e5s0, e4s0);
  wire [15:0] e7s1 = carry_save_add(e7s0, e6s0);
  wire [15:0] e9s1 = carry_save_add(e9s0, e8s0);
  wire [15:0] e11s1 = carry_save_add(e11s0, e10s0);
  wire [15:0] e13s1 = carry_save_add(e13s0, e12s0);
  wire [15:0] e15s1 = carry_save_add(e15s0, e14s0);
  wire [15:0] e17s1 = carry_save_add(e17s0, e16s0);
  wire [15:0] e19s1 = carry_save_add(e19s0, e18s0);
  wire [15:0] e21s1 = carry_save_add(e21s0, e20s0);
  wire [15:0] e23s1 = carry_save_add(e23s0, e22s0);
  wire [15:0] e25s1 = carry_save_add(e25s0, e24s0);
  wire [15:0] e27s1 = carry_save_add(e27s0, e26s0);
  wire [15:0] e29s1 = carry_save_add(e29s0, e28s0);
  wire [15:0] e31s1 = carry_save_add(e31s0, e30s0);
  wire [15:0] e33s1 = carry_save_add(e33s0, e32s0);
  wire [15:0] e35s1 = carry_save_add(e35s0, e34s0);
  wire [15:0] e37s1 = carry_save_add(e37s0, e36s0);
  wire [15:0] e39s1 = carry_save_add(e39s0, e38s0);
  wire [15:0] e41s1 = carry_save_add(e41s0, e40s0);
  wire [15:0] e43s1 = carry_save_add(e43s0, e42s0);
  wire [15:0] e45s1 = carry_save_add(e45s0, e44s0);
  wire [15:0] e47s1 = carry_save_add(e47s0, e46s0);
  wire [15:0] e49s1 = carry_save_add(e49s0, e48s0);
  wire [15:0] e51s1 = carry_save_add(e51s0, e50s0);
  wire [15:0] e53s1 = carry_save_add(e53s0, e52s0);
  wire [15:0] e55s1 = carry_save_add(e55s0, e54s0);
  wire [15:0] e57s1 = carry_save_add(e57s0, e56s0);
  wire [15:0] e59s1 = carry_save_add(e59s0, e58s0);
  wire [15:0] e61s1 = carry_save_add(e61s0, e60s0);
  wire [15:0] e63s1 = carry_save_add(e63s0, e62s0);
  wire [15:0] e3s2 = carry_save_add(e3s1, e1s1);
  wire [15:0] e7s2 = carry_save_add(e7s1, e5s1);
  wire [15:0] e11s2 = carry_save_add(e11s1, e9s1);
  wire [15:0] e15s2 = carry_save_add(e15s1, e13s1);
  wire [15:0] e19s2 = carry_save_add(e19s1, e17s1);
  wire [15:0] e23s2 = carry_save_add(e23s1, e21s1);
  wire [15:0] e27s2 = carry_save_add(e27s1, e25s1);
  wire [15:0] e31s2 = carry_save_add(e31s1, e29s1);
  wire [15:0] e35s2 = carry_save_add(e35s1, e33s1);
  wire [15:0] e39s2 = carry_save_add(e39s1, e37s1);
  wire [15:0] e43s2 = carry_save_add(e43s1, e41s1);
  wire [15:0] e47s2 = carry_save_add(e47s1, e45s1);
  wire [15:0] e51s2 = carry_save_add(e51s1, e49s1);
  wire [15:0] e55s2 = carry_save_add(e55s1, e53s1);
  wire [15:0] e59s2 = carry_save_add(e59s1, e57s1);
  wire [15:0] e63s2 = carry_save_add(e63s1, e61s1);
  wire [15:0] e7s3 = carry_save_add(e7s2, e3s2);
  wire [15:0] e15s3 = carry_save_add(e15s2, e11s2);
  wire [15:0] e23s3 = carry_save_add(e23s2, e19s2);
  wire [15:0] e31s3 = carry_save_add(e31s2, e27s2);
  wire [15:0] e39s3 = carry_save_add(e39s2, e35s2);
  wire [15:0] e47s3 = carry_save_add(e47s2, e43s2);
  wire [15:0] e55s3 = carry_save_add(e55s2, e51s2);
  wire [15:0] e63s3 = carry_save_add(e63s2, e59s2);
  wire [15:0] e15s4 = carry_save_add(e15s3, e7s3);
  wire [15:0] e31s4 = carry_save_add(e31s3, e23s3);
  wire [15:0] e47s4 = carry_save_add(e47s3, e39s3);
  wire [15:0] e63s4 = carry_save_add(e63s3, e55s3);
  wire [15:0] e31s5 = carry_save_add(e31s4, e15s4);
  wire [15:0] e63s5 = carry_save_add(e63s4, e47s4);
  wire [15:0] e63s6 = carry_save_add(e63s5, e31s5);
  // backward pass
  wire [15:0] e47s7 = carry_save_add(e47s4, e31s5);
  wire [15:0] e23s8 = carry_save_add(e23s3, e15s4);
  wire [15:0] e39s8 = carry_save_add(e39s3, e31s5);
  wire [15:0] e55s8 = carry_save_add(e55s3, e47s7);
  wire [15:0] e11s9 = carry_save_add(e11s2, e7s3);
  wire [15:0] e19s9 = carry_save_add(e19s2, e15s4);
  wire [15:0] e27s9 = carry_save_add(e27s2, e23s8);
  wire [15:0] e35s9 = carry_save_add(e35s2, e31s5);
  wire [15:0] e43s9 = carry_save_add(e43s2, e39s8);
  wire [15:0] e51s9 = carry_save_add(e51s2, e47s7);
  wire [15:0] e59s9 = carry_save_add(e59s2, e55s8);
  wire [15:0] e5s10 = carry_save_add(e5s1, e3s2);
  wire [15:0] e9s10 = carry_save_add(e9s1, e7s3);
  wire [15:0] e13s10 = carry_save_add(e13s1, e11s9);
  wire [15:0] e17s10 = carry_save_add(e17s1, e15s4);
  wire [15:0] e21s10 = carry_save_add(e21s1, e19s9);
  wire [15:0] e25s10 = carry_save_add(e25s1, e23s8);
  wire [15:0] e29s10 = carry_save_add(e29s1, e27s9);
  wire [15:0] e33s10 = carry_save_add(e33s1, e31s5);
  wire [15:0] e37s10 = carry_save_add(e37s1, e35s9);
  wire [15:0] e41s10 = carry_save_add(e41s1, e39s8);
  wire [15:0] e45s10 = carry_save_add(e45s1, e43s9);
  wire [15:0] e49s10 = carry_save_add(e49s1, e47s7);
  wire [15:0] e53s10 = carry_save_add(e53s1, e51s9);
  wire [15:0] e57s10 = carry_save_add(e57s1, e55s8);
  wire [15:0] e61s10 = carry_save_add(e61s1, e59s9);
  wire [15:0] e2s11 = carry_save_add(e2s0, e1s1);
  wire [15:0] e4s11 = carry_save_add(e4s0, e3s2);
  wire [15:0] e6s11 = carry_save_add(e6s0, e5s10);
  wire [15:0] e8s11 = carry_save_add(e8s0, e7s3);
  wire [15:0] e10s11 = carry_save_add(e10s0, e9s10);
  wire [15:0] e12s11 = carry_save_add(e12s0, e11s9);
  wire [15:0] e14s11 = carry_save_add(e14s0, e13s10);
  wire [15:0] e16s11 = carry_save_add(e16s0, e15s4);
  wire [15:0] e18s11 = carry_save_add(e18s0, e17s10);
  wire [15:0] e20s11 = carry_save_add(e20s0, e19s9);
  wire [15:0] e22s11 = carry_save_add(e22s0, e21s10);
  wire [15:0] e24s11 = carry_save_add(e24s0, e23s8);
  wire [15:0] e26s11 = carry_save_add(e26s0, e25s10);
  wire [15:0] e28s11 = carry_save_add(e28s0, e27s9);
  wire [15:0] e30s11 = carry_save_add(e30s0, e29s10);
  wire [15:0] e32s11 = carry_save_add(e32s0, e31s5);
  wire [15:0] e34s11 = carry_save_add(e34s0, e33s10);
  wire [15:0] e36s11 = carry_save_add(e36s0, e35s9);
  wire [15:0] e38s11 = carry_save_add(e38s0, e37s10);
  wire [15:0] e40s11 = carry_save_add(e40s0, e39s8);
  wire [15:0] e42s11 = carry_save_add(e42s0, e41s10);
  wire [15:0] e44s11 = carry_save_add(e44s0, e43s9);
  wire [15:0] e46s11 = carry_save_add(e46s0, e45s10);
  wire [15:0] e48s11 = carry_save_add(e48s0, e47s7);
  wire [15:0] e50s11 = carry_save_add(e50s0, e49s10);
  wire [15:0] e52s11 = carry_save_add(e52s0, e51s9);
  wire [15:0] e54s11 = carry_save_add(e54s0, e53s10);
  wire [15:0] e56s11 = carry_save_add(e56s0, e55s8);
  wire [15:0] e58s11 = carry_save_add(e58s0, e57s10);
  wire [15:0] e60s11 = carry_save_add(e60s0, e59s9);
  wire [15:0] e62s11 = carry_save_add(e62s0, e61s10);
  // outputs
  assign dout[0 +: 8] = carry_save_get(e0s0);
  assign dout[8 +: 8] = carry_save_get(e1s1);
  assign dout[16 +: 8] = carry_save_get(e2s11);
  assign dout[24 +: 8] = carry_save_get(e3s2);
  assign dout[32 +: 8] = carry_save_get(e4s11);
  assign dout[40 +: 8] = carry_save_get(e5s10);
  assign dout[48 +: 8] = carry_save_get(e6s11);
  assign dout[56 +: 8] = carry_save_get(e7s3);
  assign dout[64 +: 8] = carry_save_get(e8s11);
  assign dout[72 +: 8] = carry_save_get(e9s10);
  assign dout[80 +: 8] = carry_save_get(e10s11);
  assign dout[88 +: 8] = carry_save_get(e11s9);
  assign dout[96 +: 8] = carry_save_get(e12s11);
  assign dout[104 +: 8] = carry_save_get(e13s10);
  assign dout[112 +: 8] = carry_save_get(e14s11);
  assign dout[120 +: 8] = carry_save_get(e15s4);
  assign dout[128 +: 8] = carry_save_get(e16s11);
  assign dout[136 +: 8] = carry_save_get(e17s10);
  assign dout[144 +: 8] = carry_save_get(e18s11);
  assign dout[152 +: 8] = carry_save_get(e19s9);
  assign dout[160 +: 8] = carry_save_get(e20s11);
  assign dout[168 +: 8] = carry_save_get(e21s10);
  assign dout[176 +: 8] = carry_save_get(e22s11);
  assign dout[184 +: 8] = carry_save_get(e23s8);
  assign dout[192 +: 8] = carry_save_get(e24s11);
  assign dout[200 +: 8] = carry_save_get(e25s10);
  assign dout[208 +: 8] = carry_save_get(e26s11);
  assign dout[216 +: 8] = carry_save_get(e27s9);
  assign dout[224 +: 8] = carry_save_get(e28s11);
  assign dout[232 +: 8] = carry_save_get(e29s10);
  assign dout[240 +: 8] = carry_save_get(e30s11);
  assign dout[248 +: 8] = carry_save_get(e31s5);
  assign dout[256 +: 8] = carry_save_get(e32s11);
  assign dout[264 +: 8] = carry_save_get(e33s10);
  assign dout[272 +: 8] = carry_save_get(e34s11);
  assign dout[280 +: 8] = carry_save_get(e35s9);
  assign dout[288 +: 8] = carry_save_get(e36s11);
  assign dout[296 +: 8] = carry_save_get(e37s10);
  assign dout[304 +: 8] = carry_save_get(e38s11);
  assign dout[312 +: 8] = carry_save_get(e39s8);
  assign dout[320 +: 8] = carry_save_get(e40s11);
  assign dout[328 +: 8] = carry_save_get(e41s10);
  assign dout[336 +: 8] = carry_save_get(e42s11);
  assign dout[344 +: 8] = carry_save_get(e43s9);
  assign dout[352 +: 8] = carry_save_get(e44s11);
  assign dout[360 +: 8] = carry_save_get(e45s10);
  assign dout[368 +: 8] = carry_save_get(e46s11);
  assign dout[376 +: 8] = carry_save_get(e47s7);
  assign dout[384 +: 8] = carry_save_get(e48s11);
  assign dout[392 +: 8] = carry_save_get(e49s10);
  assign dout[400 +: 8] = carry_save_get(e50s11);
  assign dout[408 +: 8] = carry_save_get(e51s9);
  assign dout[416 +: 8] = carry_save_get(e52s11);
  assign dout[424 +: 8] = carry_save_get(e53s10);
  assign dout[432 +: 8] = carry_save_get(e54s11);
  assign dout[440 +: 8] = carry_save_get(e55s8);
  assign dout[448 +: 8] = carry_save_get(e56s11);
  assign dout[456 +: 8] = carry_save_get(e57s10);
  assign dout[464 +: 8] = carry_save_get(e58s11);
  assign dout[472 +: 8] = carry_save_get(e59s9);
  assign dout[480 +: 8] = carry_save_get(e60s11);
  assign dout[488 +: 8] = carry_save_get(e61s10);
  assign dout[496 +: 8] = carry_save_get(e62s11);
  assign dout[504 +: 8] = carry_save_get(e63s6);
endmodule

module pdep_pext_lrotc_zero #(
	parameter integer N = 1,
	parameter integer M = 1
) (
	input [7:0] din,
	output [M-1:0] dout
);
	wire [2*M-1:0] mask = {M{1'b1}};
	assign dout = (mask << din[N-1:0]) >> M;
endmodule

module pdep_pext_decoder(input [63:0] mask, output [31:0] s1, s2, s4, s8, s16, s32);
	wire [8*64-1:0] ppc_data;

	pdep_pext_ppc ppc (
		.din  (mask),
		.dout (ppc_data)
	);

	genvar i;
	generate
		for (i = 0; i < 32; i = i+1) begin:stage1
			pdep_pext_lrotc_zero #(.N(1), .M(1)) lrotc_zero (
				.din(ppc_data[8*(2*i + 1 - 1) +: 8]),
				.dout(s1[i])
			);
		end

		for (i = 0; i < 16; i = i+1) begin:stage2
			pdep_pext_lrotc_zero #(.N(2), .M(2)) lrotc_zero (
				.din(ppc_data[8*(4*i + 2 - 1) +: 8]),
				.dout(s2[2*i +: 2])
			);
		end

		for (i = 0; i < 8; i = i+1) begin:stage4
			pdep_pext_lrotc_zero #(.N(3), .M(4)) lrotc_zero (
				.din(ppc_data[8*(8*i + 4 - 1) +: 8]),
				.dout(s4[4*i +: 4])
			);
		end

		for (i = 0; i < 4; i = i+1) begin:stage8
			pdep_pext_lrotc_zero #(.N(4), .M(8)) lrotc_zero (
				.din(ppc_data[8*(16*i + 8 - 1) +: 8]),
				.dout(s8[8*i +: 8])
			);
		end

		for (i = 0; i < 2; i = i+1) begin:stage16
			pdep_pext_lrotc_zero #(.N(5), .M(16)) lrotc_zero (
				.din(ppc_data[8*(32*i + 16 - 1) +: 8]),
				.dout(s16[16*i +: 16])
			);
		end

		for (i = 0; i < 1; i = i+1) begin:stage32
			pdep_pext_lrotc_zero #(.N(6), .M(32)) lrotc_zero (
				.din(ppc_data[8*(64*i + 32 - 1) +: 8]),
				.dout(s32[32*i +: 32])
			);
		end
	endgenerate
endmodule

module pdep_pext_butterfly(input inv, input [63:0] din, input [31:0] s1, s2, s4, s8, s16, s32, output reg [63:0] dout);
	integer i;
	`define BUTTERFLY_SWAP(cond, idx, off) if (!cond) {dout[idx], dout[(idx)+(off)]} = {dout[(idx)+(off)], dout[idx]}
	always @* begin
		dout = din;

		if (inv) begin
			for (i = 0; i < 32; i = i+1)
				`BUTTERFLY_SWAP(s1[i], 2*i, 1);

			for (i = 0; i < 32; i = i+1)
				`BUTTERFLY_SWAP(s2[i], 4*(i/2) + i%2, 2);

			for (i = 0; i < 32; i = i+1)
				`BUTTERFLY_SWAP(s4[i], 8*(i/4) + i%4, 4);

			for (i = 0; i < 32; i = i+1)
				`BUTTERFLY_SWAP(s8[i], 16*(i/8) + i%8, 8);

			for (i = 0; i < 32; i = i+1)
				`BUTTERFLY_SWAP(s16[i], 32*(i/16) + i%16, 16);

			for (i = 0; i < 32; i = i+1)
				`BUTTERFLY_SWAP(s32[i], i, 32);
		end else begin
			for (i = 0; i < 32; i = i+1)
				`BUTTERFLY_SWAP(s32[i], i, 32);

			for (i = 0; i < 32; i = i+1)
				`BUTTERFLY_SWAP(s16[i], 32*(i/16) + i%16, 16);

			for (i = 0; i < 32; i = i+1)
				`BUTTERFLY_SWAP(s8[i], 16*(i/8) + i%8, 8);

			for (i = 0; i < 32; i = i+1)
				`BUTTERFLY_SWAP(s4[i], 8*(i/4) + i%4, 4);

			for (i = 0; i < 32; i = i+1)
				`BUTTERFLY_SWAP(s2[i], 4*(i/2) + i%2, 2);

			for (i = 0; i < 32; i = i+1)
				`BUTTERFLY_SWAP(s1[i], 2*i, 1);
		end
	end
	`undef BUTTERFLY_SWAP
endmodule

// -----------------------------------------------------------------------
// non-parallel multi-cycle implementations

module pdep_pext_mc (
	input clk,
	input load,
	input pext,
	output ready,
	input [63:0] din,
	input [63:0] mask,
	output [63:0] dout
);
	reg [63:0] result;
	reg [63:0] tags;

	wire [63:0] t0 = ~tags | mask;
	wire [63:0] t1 = tags + 1;
	wire [63:0] t2 = t1 ^ tags;
	wire [63:0] t3 = ~((t0 + 1) ^ t0);

	assign ready = !load && (pext ? !(t1 & tags) : !(mask & tags) || !(result & tags));
	assign dout = pext ? result : result & mask;

	always @(posedge clk) begin
		if (load) begin
			result <= pext ? din & mask : din;
			tags <= pext ? mask : ~0;
		end else
		if (pext) begin
			result <= ((result & ~t2) >> 1) | (result & t2);
			tags <= ((tags & ~t2) >> 1) | (tags & t2);
		end else begin
			result <= ((result << 1 ) & t3) | (result & ~t3);
			tags <= t3;
		end
	end
endmodule

module pdep_pext_mc2 (
	input clk,
	input load,
	input pext,
	output reg ready,
	input [63:0] din,
	input [63:0] mask,
	output [63:0] dout
);
	reg [5:0] cnt;
	reg [63:0] result;
	reg [63:0] tags;

	assign dout = pext ? result : result & mask;

	wire [63:0] cursor = (tags << 1) ^ tags;

	always @(posedge clk) begin
		cnt <= cnt + !ready;
		ready <= !load && &cnt;

		if (load) begin
			result <= pext ? din & mask : din;
			tags <= pext ? mask : ~0;
			cnt <= 0;
		end else
		if (pext) begin
			result <= (((result & tags) >> 1) & ~tags) | ((result & tags) & {tags, 1'b1});
			tags <= ((tags >> 1) & ~tags) | (tags & {tags, 1'b1});
		end else begin
			if (!(cursor & mask))
				result <= ((result & tags) << 1) | (result & ~tags);
			tags <= tags << 1;
		end
	end
endmodule

module pdep_pext_mc3 (
	input clk,
	input load,
	input pext,
	output ready,
	input [63:0] din,
	input [63:0] mask,
	output [63:0] dout
);
	reg [63:0] c, m, msk;
	wire [63:0] b = msk & -msk;

	assign ready = !msk && !load;
	assign dout = c;

	always @(posedge clk) begin
		if (load) begin
			c <= 0;
			m <= 1;
			msk <= mask;
		end else
		if (!ready) begin
			if (din & (pext ? b : m))
				c <= c | (pext ? m : b);
			msk <= msk & ~b;
			m <= m << 1;
		end
	end
endmodule

// -----------------------------------------------------------------------
// a few simple modules for size comparison

module shift64 (input [2:0] mode, input [63:0] din, input [5:0] shamt, output [63:0] dout);
	wire [63:0] pad =
			mode[2:1] == 2'b00 ? 64'b0 :
			mode[2:1] == 2'b01 ? ~64'b0 :
			mode[2:1] == 2'b10 ? {64{din[63]}} : din;
	assign dout = mode[0] ? {pad, din} >> shamt : ({din, pad} << shamt) >> 64;
endmodule

module rshift64 (input [63:0] din, input [5:0] shamt, output [63:0] dout);
	assign dout = din >> shamt;
endmodule

module mul32 (input [31:0] din0, din1, output [63:0] dout);
	assign dout = din0 * din1;
endmodule

module mul16 (input [15:0] din0, din1, output [31:0] dout);
	assign dout = din0 * din1;
endmodule

module grev(input [63:0] din, input s1, s2, s4, s8, s16, s32, output reg [63:0] dout);
	integer i;
	`define BUTTERFLY_SWAP(cond, idx, off) if (!cond) {dout[idx], dout[(idx)+(off)]} = {dout[(idx)+(off)], dout[idx]}
	always @* begin
		dout = din;

		for (i = 0; i < 32; i = i+1)
			`BUTTERFLY_SWAP(s32, i, 32);

		for (i = 0; i < 32; i = i+1)
			`BUTTERFLY_SWAP(s16, 32*(i/16) + i%16, 16);

		for (i = 0; i < 32; i = i+1)
			`BUTTERFLY_SWAP(s8, 16*(i/8) + i%8, 8);

		for (i = 0; i < 32; i = i+1)
			`BUTTERFLY_SWAP(s4, 8*(i/4) + i%4, 4);

		for (i = 0; i < 32; i = i+1)
			`BUTTERFLY_SWAP(s2, 4*(i/2) + i%2, 2);

		for (i = 0; i < 32; i = i+1)
			`BUTTERFLY_SWAP(s1, 2*i, 1);
	end
	`undef BUTTERFLY_SWAP
endmodule

// -----------------------------------------------------------------------
// top modules with low pin count for FPGA test synthesis

module pdep_pext_syntop(input clk, input sample, pdep, pext, input [1:0] shift_in, output shift_out);
	reg [63:0] din, mask, dout_reg;
	wire [63:0] dout;

`ifdef PDEP_PEXT_PIPELINE
	pdep_pext uut (clk, pdep, pext, din, mask, dout);
`else
	pdep_pext uut (pdep, pext, din, mask, dout);
`endif

	always @(posedge clk) begin
		dout_reg <= sample ? dout : dout_reg >> 1;
		din <= {din, shift_in[0]};
		mask <= {mask, shift_in[1]};
	end

	assign shift_out = dout_reg[0];
endmodule

module pdep_pext_mc_syntop(input clk, input sample, load, pext, input [1:0] shift_in, output ready, output shift_out);
	reg [63:0] din, mask, dout_reg;
	wire [63:0] dout;

	pdep_pext_mc uut (clk, load, pext, ready, din, mask, dout);

	always @(posedge clk) begin
		dout_reg <= sample ? dout : dout_reg >> 1;
		din <= {din, shift_in[0]};
		mask <= {mask, shift_in[1]};
	end

	assign shift_out = dout_reg[0];
endmodule

module pdep_pext_mc2_syntop(input clk, input sample, load, pext, input [1:0] shift_in, output ready, output shift_out);
	reg [63:0] din, mask, dout_reg;
	wire [63:0] dout;

	pdep_pext_mc2 uut (clk, load, pext, ready, din, mask, dout);

	always @(posedge clk) begin
		dout_reg <= sample ? dout : dout_reg >> 1;
		din <= {din, shift_in[0]};
		mask <= {mask, shift_in[1]};
	end

	assign shift_out = dout_reg[0];
endmodule

