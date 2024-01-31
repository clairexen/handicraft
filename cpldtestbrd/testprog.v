
module top(
	sw1, sw2, sw3, sw4,
	led1, led2, led3, led4,
	gck1, gck2, gck3, gts1, gts2, gsr,
	p1_1, p1_2, p1_4, p1_6, p1_8, p1_9, p1_10, p1_11,
	p2_1, p2_2, p2_4, p2_7, p2_8, p2_9, p2_10, p2_11
);

input sw1, sw2, sw3, sw4;
output led1, led2, led3, led4;
input gck1, gck2, gck3, gts1, gts2, gsr;
inout p1_1, p1_2, p1_4, p1_6, p1_8, p1_9, p1_10, p1_11;
inout p2_1, p2_2, p2_4, p2_7, p2_8, p2_9, p2_10, p2_11;

assign led1 = !sw1;
assign led2 = !sw2;
assign led3 = !sw3;
assign led4 = !sw4;

// wire [3:0] in;
// reg [3:0] out;
// wire mode, clk;
// 
// assign in = { !sw4, !sw3, !sw2, !sw1 };
// assign { led4, led3, led2, led1 } = out;
// assign mode = p2_10;
// assign clk = p2_11;
// 
// always @(posedge clk)
// 	out <= mode ? in : in + out;

endmodule

