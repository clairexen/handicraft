
// helper module for floating gates

module KEEP(output reg y, input x);
always @* if (x === 0 || x === 1) y <= x; else if (y === 1'bx) y <= 0;
endmodule

module LATCH(output reg y, input x, input en);
wire x_buf, en_buf;
KEEP keep_x(x_buf, x);
KEEP keep_en(en_buf, en);
always @* if (en_buf) y <= x_buf;
endmodule

// simulation models for cells from nmos.v

module PULLUP(output y);
wire vcc = 1;
buf (weak1, weak0) g1 (y, vcc);
endmodule

module SW(input gate, inout cc1, inout cc2);
wire gate_buf;
KEEP keep_gate (gate_buf, gate);
tranif1 sw (cc1, cc2, gate_buf);
endmodule

module SW0(input gate, output cc);
wire gate_buf;
KEEP keep_gate (gate_buf, gate);
assign cc = gate_buf ? 1'b0 : 1'bz;
endmodule

module SW1(input gate, output cc);
wire gate_buf;
KEEP keep_gate (gate_buf, gate);
assign cc = gate_buf ? 1'b1 : 1'bz;
endmodule

module gate_not(input a, output y);
KEEP keep(y, ~a);
endmodule

module gate_nor2(input a, input b, output y);
KEEP keep(y, ~(a|b));
endmodule

module gate_nor3(input a, input b, input c, output y);
KEEP keep(y, ~(a|b|c));
endmodule

module gate_nor4(input a, input b, input c, input d, output y);
KEEP keep(y, ~(a|b|c|d));
endmodule

module gate_nor5(input a, input b, input c, input d, input e, output y);
KEEP keep(y, ~(a|b|c|d|e));
endmodule

module gate_nor6(input a, input b, input c, input d, input e, input f, output y);
KEEP keep(y, ~(a|b|c|d|e|f));
endmodule

module gate_nor7(input a, input b, input c, input d, input e, input f, input g, output y);
KEEP keep(y, ~(a|b|c|d|e|f|g));
endmodule

module gate_nor8(input a, input b, input c, input d, input e, input f, input g, input h, output y);
KEEP keep(y, ~(a|b|c|d|e|f|g|h));
endmodule

module gate_nor9(input a, input b, input c, input d, input e, input f, input g, input h, input i, output y);
KEEP keep(y, ~(a|b|c|d|e|f|g|h|i));
endmodule

module gate_nand2(input a, input b, output y);
KEEP keep(y, ~(a&b));
endmodule

module gate_nand3(input a, input b, input c, output y);
KEEP keep(y, ~(a&b&c));
endmodule

// simulation models for cells from cells_stage0.v

module dlatch_not(input en, input a, output y, output q);
KEEP keep(y, ~a);
LATCH latch(q, y, en);
endmodule

module dlatch_nor2(input en, input a, input b, output y, output q);
KEEP keep(y, ~(a|b));
LATCH latch(q, y, en);
endmodule

module dlatch_nor3(input en, input a, input b, input c, output y, output q);
KEEP keep(y, ~(a|b|c));
LATCH latch(q, y, en);
endmodule

module dlatch_nor4(input en, input a, input b, input c, input d, output y, output q);
KEEP keep(y, ~(a|b|c|d));
LATCH latch(q, y, en);
endmodule

module buffer(input en0, input en1, output q);
wire en0_buf, en1_buf;
KEEP keep_en0(en0_buf, en0);
KEEP keep_en1(en1_buf, en1);
assign q = en0_buf ? 1'b0 : en1_buf ? 1'b1 : 1'bz;
endmodule

// simulation models for cells from cells_stage1.v

module clock_generator(input clk0, output reg clk1out, output reg clk2out, output reg cclk, output reg cp1);
always @(posedge clk0) begin
	clk1out <= 0;
	clk2out <= #1 1;
end
always @(negedge clk0) begin
	clk2out <= 0;
	clk1out <= #1 1;
end
always @*
	cp1 <= clk1out;
always @*
	cclk <= clk2out;
endmodule

