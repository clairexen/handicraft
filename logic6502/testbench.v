
module testbench;

wire vss, rdy, clk1out, irq, nmi, sync, vcc, ab0, ab1, ab2, ab3, ab4, ab5, ab6, ab7, ab8, ab9, ab10, ab11;
wire ab12, ab13, ab14, ab15, db7, db6, db5, db4, db3, db2, db1, db0, rw, clk0, so, clk2out, res;

MOS6502 uut (
	.vss(vss), .rdy(rdy), .clk1out(clk1out), .irq(irq), .nmi(nmi), .sync(sync), .vcc(vcc),
	.ab0(ab0), .ab1(ab1), .ab2(ab2), .ab3(ab3), .ab4(ab4), .ab5(ab5), .ab6(ab6), .ab7(ab7),
	.ab8(ab8), .ab9(ab9), .ab10(ab10), .ab11(ab11), .ab12(ab12), .ab13(ab13), .ab14(ab14), .ab15(ab15),
	.db7(db7), .db6(db6), .db5(db5), .db4(db4), .db3(db3), .db2(db2), .db1(db1), .db0(db0),
	.rw(rw), .clk0(clk0), .so(so), .clk2out(clk2out), .res(res)
);

initial begin
	$dumpfile("testbench.vcd");
	$dumpvars(0, testbench);
end

// clock and reset

reg clk = 0, reset = 0;
reg clk1 = 1, clk2 = 0;

assign clk0 = clk, res = reset;
assign clk1out = clk1, clk2out = clk2;

always @(posedge clk) begin
	clk1 <= 0;
	clk2 <= #10 1;
end

always @(negedge clk) begin
	clk2 <= 0;
	clk1 <= #10 1;
end

initial begin
	clk <= 0; #150;
	forever begin
		#50;
		clk <= ~clk;
	end
end

initial begin
	// @(posedge clk);
	#20;
	$display($time);
`include "cmpnets.v"
end

initial begin
	reset <= 0;
	#1995;
	reset <= 1;
end

initial begin
	#10000;
	$finish;
end

// memory

// timing diagrams:
// http://www.geocities.co.jp/SiliconValley-Bay/9975/PC2E/english.html#6502timing

/*
reg [7:0] mem [0:511];
integer k;

initial begin
	for (k = 0; k < 512; k = k+1)
		mem[k] = 0;

	mem[16'h0000] = 8'h a9;
	mem[16'h0001] = 8'h 00;
	mem[16'h0002] = 8'h 20;
	mem[16'h0003] = 8'h 10;
	mem[16'h0004] = 8'h 00;
	mem[16'h0005] = 8'h 4c;
	mem[16'h0006] = 8'h 02;
	mem[16'h0007] = 8'h 00;

	mem[16'h0008] = 8'h 00;
	mem[16'h0009] = 8'h 00;
	mem[16'h000a] = 8'h 00;
	mem[16'h000b] = 8'h 00;
	mem[16'h000c] = 8'h 00;
	mem[16'h000d] = 8'h 00;
	mem[16'h000e] = 8'h 00;
	mem[16'h000f] = 8'h 40;

	mem[16'h0010] = 8'h e8;
	mem[16'h0011] = 8'h 88;
	mem[16'h0012] = 8'h e6;
	mem[16'h0013] = 8'h 0f;
	mem[16'h0014] = 8'h 38;
	mem[16'h0015] = 8'h 69;
	mem[16'h0016] = 8'h 02;
	mem[16'h0017] = 8'h 60;
end
*/

wire [15:0] ab = { ab15, ab14, ab13, ab12, ab11, ab10, ab9, ab8, ab7, ab6, ab5, ab4, ab3, ab2, ab1, ab0 };
wire [7:0] db = { db7, db6, db5, db4, db3, db2, db1, db0 };

/*
always @*
	if (!rw && res && ab < 512)
		mem[ab] <= db;

assign { db7, db6, db5, db4, db3, db2, db1, db0 } = rw ? ab < 512 ? mem[ab] : 8'hff : 8'bz;
*/

assign db = rw ? 8'hEA : 8'bz; // NOP

// other inputs

assign irq = 1, nmi = 1, so = 1, rdy = 1;

endmodule


module SR (S, R, Q);
input S, R;
output reg Q = 0;
always @* begin
	if (R)
		Q <= 0;
	else
	if (S)
		Q <= 1;
end
endmodule

module SW0 (input gate, inout cc);
assign cc = gate ? 1'b0 : 1'bz;
endmodule

module SW1 (input gate, inout cc);
assign cc = gate ? 1'b0 : 1'bz;
endmodule

module PULLUP (input y);
/* just ignore remaining pullups */
endmodule

