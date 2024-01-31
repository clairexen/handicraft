module top(output [7:0] a, b);
	localparam [15:0] adata = "a\000";
	localparam [15:0] bdata = "b\001";
	assign a = adata[15:8], b = bdata[15:8];
endmodule
