/*
 *  Copyright (C) 2020  Claire Wolf <claire@symbioticeda.com>
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

module miter (
	input        clock,
	input        reset,
	input  [7:0] din_di1,
	input  [7:0] din_di2,
	input  [1:0] din_fun,
	input        din_vld
);
	wire       ref_din_rdy;
	wire [7:0] ref_dout_dat;
	wire       ref_dout_vld;

	wire       uut_din_rdy;
	wire [7:0] uut_dout_dat;
	wire       uut_dout_vld;

	serial_alu ref (
		.mutsel  (1'b 0),
		.clock   (clock   ),
		.reset   (reset   ),
		.din_di1 (din_di1 ),
		.din_di2 (din_di2 ),
		.din_fun (din_fun ),
		.din_vld (din_vld ),
		.din_rdy (ref_din_rdy ),
		.dout_dat(ref_dout_dat),
		.dout_vld(ref_dout_vld)
	);

	serial_alu uut (
		.mutsel  (1'b 1),
		.clock   (clock   ),
		.reset   (reset   ),
		.din_di1 (din_di1 ),
		.din_di2 (din_di2 ),
		.din_fun (din_fun ),
		.din_vld (din_vld ),
		.din_rdy (uut_din_rdy ),
		.dout_dat(uut_dout_dat),
		.dout_vld(uut_dout_vld)
	);

	initial assume (reset);

	always @* begin
		if (!reset) begin
			assert (ref_din_rdy == uut_din_rdy);
			assert (ref_dout_vld == uut_dout_vld);
			if (ref_dout_vld)
				assert (ref_dout_dat == uut_dout_dat);
		end
	end
endmodule
