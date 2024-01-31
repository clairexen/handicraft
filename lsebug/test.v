`timescale 1 ns / 1 ps

module test_a (
	input clk, resetn,
	input [1:0] data_bits,
	input [1:0] data_enable,
	output reg [1:0] serdes_out
);
	reg [2:0] queue_bits;
	reg [2:0] queue_enable;
	reg shift;

	always @(posedge clk) begin
		if (!resetn) begin
			queue_enable = 0;
			serdes_out <= 0;
			shift <= 0;
		end else begin
			if (shift) begin
				if (queue_enable[0]) begin
					queue_bits = {data_bits, queue_bits[0]};
					queue_enable = {data_enable, queue_enable[0]};
				end else begin
					queue_bits = data_bits;
					queue_enable = data_enable;
				end
			end

			serdes_out <= queue_bits[1:0] & queue_enable[1:0];

			queue_bits = queue_bits >> 2;
			queue_enable = queue_enable >> 2;

			shift <= !queue_enable[1];
		end
	end
endmodule

module test_b (
	input clk, resetn,
	input [1:0] data_bits,
	input [1:0] data_enable,
	output reg [1:0] serdes_out
);
	reg [2:0] nxt_serdes_out;
	reg [2:0] nxt_queue_bits;
	reg [2:0] nxt_queue_enable;
	reg nxt_shift;

	reg [2:0] queue_bits;
	reg [2:0] queue_enable;
	reg shift;

	always @* begin
		nxt_queue_bits = queue_bits;
		nxt_queue_enable = queue_enable;
		nxt_serdes_out = serdes_out;
		nxt_shift = 0;

		if (shift) begin
			if (queue_enable[0]) begin
				nxt_queue_bits = {data_bits, queue_bits[0]};
				nxt_queue_enable = {data_enable, queue_enable[0]};
			end else begin
				nxt_queue_bits = data_bits;
				nxt_queue_enable = data_enable;
			end
		end

		nxt_serdes_out = nxt_queue_bits[1:0] & nxt_queue_enable[1:0];

		nxt_queue_bits = nxt_queue_bits >> 2;
		nxt_queue_enable = nxt_queue_enable >> 2;
		nxt_shift = !nxt_queue_enable[1];
	end

	always @(posedge clk) begin
		if (!resetn) begin
			serdes_out <= 0;
			queue_bits <= 0;
			queue_enable <= 0;
			shift <= 0;
		end else begin
			serdes_out <= nxt_serdes_out;
			queue_bits <= nxt_queue_bits;
			queue_enable <= nxt_queue_enable;
			shift <= nxt_shift;
		end
	end
endmodule

module test_c (
	input clk, resetn,
	input [1:0] data_bits,
	input [1:0] data_enable,
	output reg [1:0] serdes_out
);
	reg resetn_q;

	always @(posedge clk) begin
		resetn_q <= resetn;
		serdes_out <= data_bits & data_enable & {2{resetn && resetn_q}};
	end
endmodule

