module test;
	reg foo = 0;
	wire dout, ck;
	demo uut (foo, dout, ck);

	initial begin
		$dumpfile("test.vcd");
		$dumpvars(0, test);
		$monitor("foo=%b, dout=%b, ck=%b", foo, dout, ck);
		repeat (10) begin
			foo = !foo;
			#10;
		end
	end
endmodule

