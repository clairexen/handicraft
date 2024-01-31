// for x in {0..9}; do echo $x$x > fc2_$x.hex; done
// yosys -p 'synth_ice40 -top test' vlhexgenloop.v
module test (
	input              clk, // clock
	input  [     8:0] addr, // address
	output [8*10-1:0] coef  // values
);
	function [7:0] ascii_digit;
		input [3:0] value;
		begin
			ascii_digit = "0" + value;
		end
	endfunction

	genvar k;
	generate
		for (k = 0; k < 10; k = k + 1) begin:K
			rom_block #({"fc2_", ascii_digit(k), ".hex"}) m (clk, addr, coef[8*k+7:8*k]);
		end
	endgenerate
endmodule

module rom_block (
	input clk,             // clock
	input      [8:0] addr, // address
	output reg [7:0] coef  // values
);
	parameter hexfile = "";
	reg [7:0] romdata [0:511];

	initial
		if (hexfile) $readmemh(hexfile, romdata);

	always @(posedge clk)
		coef <= romdata[addr];
endmodule
