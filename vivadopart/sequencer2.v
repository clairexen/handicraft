module sequencer2 (
	input [4:0] seqidx,
	output reg [7:0] leds
);
	always @* begin
		leds = 'bx;
		case (seqidx)
			'h00: leds = 8'b 10001000;
			'h01: leds = 8'b 01000100;
			'h02: leds = 8'b 00100010;
			'h03: leds = 8'b 00010001;
			'h04: leds = 8'b 10001000;
			'h05: leds = 8'b 01000100;
			'h06: leds = 8'b 00100010;
			'h07: leds = 8'b 00010001;
			'h08: leds = 8'b 10001000;
			'h09: leds = 8'b 01000100;
			'h0a: leds = 8'b 00100010;
			'h0b: leds = 8'b 00010001;
			'h0c: leds = 8'b 10001000;
			'h0d: leds = 8'b 01000100;
			'h0e: leds = 8'b 00100010;
			'h0f: leds = 8'b 00010001;
			'h10: leds = 8'b 10001000;
			'h11: leds = 8'b 01000100;
			'h12: leds = 8'b 00100010;
			'h13: leds = 8'b 00010001;
			'h14: leds = 8'b 10001000;
			'h15: leds = 8'b 01000100;
			'h16: leds = 8'b 00100010;
			'h17: leds = 8'b 00010001;
			'h18: leds = 8'b 10001000;
			'h19: leds = 8'b 01000100;
			'h1a: leds = 8'b 00100010;
			'h1b: leds = 8'b 00010001;
			'h1c: leds = 8'b 10001000;
			'h1d: leds = 8'b 01000100;
			'h1e: leds = 8'b 00100010;
			'h1f: leds = 8'b 00010001;
		endcase
	end
endmodule
