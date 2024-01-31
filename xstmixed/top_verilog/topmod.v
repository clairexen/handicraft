
module topmod(
        sw1, sw2, sw3, sw4,
        led1, led2, led3, led4
);

input sw1, sw2, sw3, sw4;
output led1, led2, led3, led4;

wire [2:0] in_data;
wire [2:0] out_data;
wire en_data;

assign in_data = {!sw4, !sw3, !sw2};
assign {led4, led3, led2} = out_data;
assign en_data = !sw1;

regcell regcell(
	.\input (in_data),
	.enable(en_data),
	.Q(out_data)
);

assign led1 = !sw1;

endmodule

