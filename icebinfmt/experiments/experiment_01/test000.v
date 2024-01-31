module top(input alice, output bob);
	(* syn_noprune *)
	SB_LUT4 #(
		.LUT_INIT(16'b1010_1010_1010_1010)
	) carol (
		.I0(alice),
		.I1(1'b0),
		.I2(1'b0),
		.I3(1'b0),
		.O(bob)
	);
endmodule
