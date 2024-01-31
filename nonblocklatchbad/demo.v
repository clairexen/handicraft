module demo (input foo, output reg dout, output ck);
	wire en = 1;
	reg bar;
	initial dout = 0;
	assign ck = bar ^ foo;

	always @*
		if (en) bar <= foo;

	always @(posedge ck)
		dout <= !dout;
endmodule
