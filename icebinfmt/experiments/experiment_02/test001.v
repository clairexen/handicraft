module top(input a, output y);
	(* syn_noprune *)
	SB_LUT4 #(
		.LUT_INIT(16'b1101001011011101)
	) carol (
		.I0(1'b0),
		.I1(a),
		.I2(1'b0),
		.I3(1'b0),
		.O(y)
	);
endmodule
