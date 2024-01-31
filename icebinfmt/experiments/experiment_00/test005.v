module top(output bob);
	(* syn_noprune *)
	SB_LUT4 #(
		.LUT_INIT(16'b0000_0000_0000_0000)
	) alice (
		.I0(1'b0),
		.I1(bob),
		.I2(1'b0),
		.I3(1'b0),
		.O(bob)
	);
endmodule
