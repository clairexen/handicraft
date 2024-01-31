module top(
	input clk,
	output [7:0] leds,
	output reg [7:0] seg,
	output reg [3:0] an
);
	localparam BITS = 5;
	localparam LOG2DELAY = 22;

	reg [BITS+LOG2DELAY-1:0] counter = 0;
	reg [BITS-1:0] outcnt;

	always @(posedge clk) begin
		counter <= counter + 1;
		outcnt <= counter >> LOG2DELAY;
	end

	sequencer seq_inst (
		.seqidx(outcnt),
		.leds(leds)
	);

	// Simple 7-segment Counter

	reg [20:0] segs_counter = 0;

	wire digit_count = &segs_counter;
	wire [1:0] segs_digidx = segs_counter >> 14;
	wire [3:0] segs_phase = segs_counter >> 10;

	reg [3:0] digit3 = 0, digit2 = 0, digit1 = 0, digit0 = 0;

	wire [3:0] digit =
			segs_digidx == 3 ? digit3 :
			segs_digidx == 2 ? digit2 :
			segs_digidx == 1 ? digit1 : digit0;

	always @(posedge clk) begin
		segs_counter <= segs_counter + 1;

		if (digit_count) begin
			if (digit3 == 9 && digit2 == 9 && digit1 == 9 && digit0 == 9) begin
				digit3 <= 0;
				digit2 <= 0;
				digit1 <= 0;
				digit0 <= 0;
			end else
			if (digit2 == 9 && digit1 == 9 && digit0 == 9) begin
				digit3 <= digit3 + 1;
				digit2 <= 0;
				digit1 <= 0;
				digit0 <= 0;
			end else
			if (digit1 == 9 && digit0 == 9) begin
				digit2 <= digit2 + 1;
				digit1 <= 0;
				digit0 <= 0;
			end else
			if (digit0 == 9) begin
				digit1 <= digit1 + 1;
				digit0 <= 0;
			end else begin
				digit0 <= digit0 + 1;
			end
		end

		if ((&segs_phase) || (!segs_phase)) begin
			seg <= ~0;
			an <= ~0;
		end else begin
			case (digit)
				//           DP GFEDCBA
				0: seg <= ~8'b0_0111111;
				1: seg <= ~8'b0_0000110;
				2: seg <= ~8'b0_1011011;
				3: seg <= ~8'b0_1001111;
				4: seg <= ~8'b0_1100110;
				5: seg <= ~8'b0_1101101;
				6: seg <= ~8'b0_1111101;
				7: seg <= ~8'b0_0000111;
				8: seg <= ~8'b0_1111111;
				9: seg <= ~8'b0_1101111;
			endcase
			an <= ~(1 << segs_digidx);
		end
	end
endmodule

module sequencer (
	input [4:0] seqidx,
	output reg [7:0] leds
);
endmodule
