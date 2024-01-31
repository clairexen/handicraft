
module clock_generator(clk0, clk1out, clk2out, cclk, cp1);

// clk1out and cp1 are identical
// clk2out and cclk are identical
// cp1 and cclk are non-overlapping
// http://forum.6502.org/viewtopic.php?f=1&t=2412

input clk0;
output clk1out, clk2out, cclk, cp1;

wire n358, n519, n1129, n1715, n127, n670;
wire n1467, n1399, n135, n747, n1105, n1417;

gate_not _12_ (.a(clk0), .y(n358));
gate_not _13_ (.a(clk0), .y(n519));

gate_nor2 _15_ (.a(cp1), .b(n358), .y(n1129));
gate_not _11_ (.a(n358), .y(n1715));
gate_nor2 _9_ (.a(cp1), .b(n519), .y(n127));
gate_not _14_ (.a(n519), .y(n670));

buffer _2_ (.en0(n1129), .en1(n358), .q(n1467));
gate_not _10_ (.a(n1715), .y(n1399));
buffer _1_ (.en0(n127), .en1(n519), .q(n135));
gate_not _16_ (.a(n670), .y(n747));

buffer _8_ (.en0(n1467), .en1(n1129), .q(cclk));
buffer _7_ (.en0(n1399), .en1(n1715), .q(n1105));
buffer _4_ (.en0(n135), .en1(n127), .q(clk2out));
buffer _6_ (.en0(n747), .en1(n670), .q(n1417));

buffer _3_ (.en0(n1105), .en1(n1399), .q(cp1));
buffer _5_ (.en0(n1417), .en1(n747), .q(clk1out));

endmodule

// -------------------------------------------------------


