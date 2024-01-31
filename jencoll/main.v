module main (
	input             clock,
	output            done,

	input             sample_1,
	input      [ 7:0] value_1,
	output reg [31:0] hash_1,

	input             sample_2,
	input      [ 7:0] value_2,
	output reg [31:0] hash_2
);
	reg init_cycle = 1;

	reg last_sample_1 = 0;
	reg last_sample_2 = 0;
	reg diff_values = 0;

	reg done_1 = 0;
	reg done_2 = 0;
	wire done;

	wire j1_complete, j2_complete;
	wire [31:0] j1_hash, j2_hash;

	assign done = done_1 && done_2;

	jenkins j1 (
		.CLOCK(clock),
		.sample(sample_1),
		.value(value_1),
		.hash(j1_hash),
		.complete(j1_complete)
	);

	jenkins j2 (
		.CLOCK(clock),
		.sample(sample_2),
		.value(value_2),
		.hash(j2_hash),
		.complete(j2_complete)
	);

	always @(posedge clock) begin
		if (sample_1 != sample_2)
			diff_values <= 1;
		if (sample_1 && sample_2 && value_1 != value_2)
			diff_values <= 1;

		if (sample_1) assume (
			("a" <= value_1 && value_1 <= "z") ||
			("A" <= value_1 && value_1 <= "Z") ||
			("0" <= value_1 && value_1 <= "9")
		);

		if (sample_2) assume (
			("a" <= value_2 && value_2 <= "z") ||
			("A" <= value_2 && value_2 <= "Z") ||
			("0" <= value_2 && value_2 <= "9")
		);

		if (!init_cycle) begin
			if (j1_complete) begin
				hash_1 <= j1_hash;
				done_1 <= 1;
			end
			if (j2_complete) begin
				hash_2 <= j2_hash;
				done_2 <= 1;
			end
		end

		if (done) begin
			assume(diff_values);
			assert(hash_1 != hash_2);
		end

		init_cycle <= 0;
		last_sample_1 <= sample_1 || init_cycle;
		last_sample_2 <= sample_2 || init_cycle;

		assume(last_sample_1 || !sample_1);
		assume(last_sample_2 || !sample_2);
	end
endmodule
