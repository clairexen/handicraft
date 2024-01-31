module demo (input clk, input ce);
	reg [7:0] lfsr_1 = 1;
	reg [7:0] lfsr_2 = 1;

	always @(posedge clk) begin
		if (ce) begin
			lfsr_1 <= {^(lfsr_1 & 8'h2d), lfsr_1[7:1]};
			lfsr_2 <= (lfsr_2 >> 1) | (^(~lfsr_1 | ~8'h2d) << 7);
		end
		assert(lfsr_1[0] == lfsr_2[0]);

		// Fix #1: Fairness on CE
		//assume(ce || $past(ce) || $past(ce, 2) || $past(ce, 3));

		// Fix #2: Assert additional invariants
		//assert(lfsr_1 == lfsr_2);

		// Fix #3: Solve with PDR and let PDR discover additional invariants
	end
endmodule
