/*
 *  Copyright (C) 2019  Clifford Wolf <clifford@clifford.at>
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
 *
 *  gcc -Werror -Wall -Wextra -O2 -o short2float short2float.c && ./short2float
 *  iverilog -D TESTBENCH -o short2float.vvp short2float.v && vvp -N short2float.vvp
 *
 */

module short2float (
	input [15:0] din,
	output [31:0] dout
);
	reg [6:0] e;
	reg [15:0] m;

	always @* begin
		e = {7{|din}};
		m = din[15] ? -din : din;
		if (!m[15: 8]) begin e[3] = 0; m = m << 8; end
		if (!m[15:12]) begin e[2] = 0; m = m << 4; end
		if (!m[15:14]) begin e[1] = 0; m = m << 2; end
		if (!m[15:15]) begin e[0] = 0; m = m << 1; end
	end

	assign dout = {din[15], 1'd 0, e, m[14:1], 9'd 0};
endmodule

`ifdef TESTBENCH
module testbench;
	integer i;
	reg [31:0] testdata [0:65535];
	reg [15:0] din;
	wire [31:0] dout;

	short2float uut (din, dout);

	initial begin
		$dumpfile("short2float.vcd");
		$dumpvars(0, testbench);
		$readmemh("short2float.hex", testdata);

		for (i = 0; i < 65536; i = i+1) begin
			din = i-32768;
			#5;
			$display("%x %x %x", din, dout, testdata[i]);
			if (dout != testdata[i]) begin
				$display("ERROR");
				$stop;
			end
			#5;
		end

		$display("OKAY");
		$finish;
	end
endmodule
`endif
