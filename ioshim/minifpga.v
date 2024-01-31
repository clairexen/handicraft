module minifpga_tile (
	input clk,
	input [127:0] glbsigs,
	input [31:0] chainsigs,
	output [7:0] outsigs,
	input [5*16+8*33-1:0] cfgbits
);
	wire [15:0] locsigs;

	genvar i, j;
	generate for (i = 0; i < 16; i=i+1) begin:locmux
		wire [31:0] ins;
		wire [4:0] cfg = cfgbits[5*i +: 5];
		for (j = 0; j < 32; j=j+1)
			assign ins[j] = glbsigs[(4*j+i) % 128];
		assign locsigs[i] = ins[cfg];
	end endgenerate

	generate for (i = 0; i < 8; i = i+1) begin:luts
		wire [32:0] cfg = cfgbits[5*16 + i*33 +: 33];
		wire [15:0] lutinit = cfg[16 +: 16];
		wire bypass_ff = cfg[32];
		wire [3:0] lutin;
		for (j = 0; j < 4; j=j+1) begin:lutinsel
			reg [15:0] lutinsigs;
			always @* begin
				lutinsigs = locsigs;
				lutinsigs[(4*i+j) % 16] = chainsigs[4*i+j];
			end
			assign lutin[j] = locsigs[cfg[4*j +: 4]];
		end
		wire lutout = lutinit[lutin];
		reg lutout_reg;
		always @(posedge clk)
			lutout_reg <= lutout;
		assign outsigs[i] = bypass_ff ? lutout : lutout_reg;
	end endgenerate
endmodule
