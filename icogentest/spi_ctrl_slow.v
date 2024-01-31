`timescale 1 ns / 1 ps

module spi_ctrl_slow #(
	parameter NUM_ENDPOINTS = 3
) (
	input clk,
	input resetn,

	input       spi_sclk,
	input       spi_mosi,
	output reg  spi_miso,
	input [7:0] spi_csel,

	output reg spi_ctrl_si,
	output reg spi_ctrl_so,
	output reg spi_ctrl_hd,
	output reg [7:0] spi_ctrl_di,
	input  [7:0] spi_ctrl_do,

	output reg [NUM_ENDPOINTS-1:0] epsel
);
	// --------------------------
	// epsel decoder

	reg cselected;
	reg [7:0] spi_csel_mask;
	reg [7:0] spi_csel_masked_1;
	reg [7:0] spi_csel_masked_2;
	reg [7:0] spi_csel_masked_3;

	integer i;

	initial begin
		spi_csel_mask = 0;
		for (i = 0; i < NUM_ENDPOINTS; i = i+1)
			spi_csel_mask = spi_csel_mask | (i+1);
	end

	always @(posedge clk) begin
		spi_csel_masked_1 <= spi_csel & spi_csel_mask;
		spi_csel_masked_2 <= spi_csel_masked_1;
		spi_csel_masked_3 <= spi_csel_masked_2;
		cselected <= spi_csel_masked_3 == spi_csel_masked_2;
	end

	always @(posedge clk) begin
		epsel <= 0;
		for (i = 0; i < NUM_ENDPOINTS; i = i+1)
			if (spi_csel_masked_3 == i+1) epsel[i] <= 1;
	end


	// --------------------------
	// mosi/miso buffers

	wire buf_miso;
	reg buf_mosi;

	always @(negedge spi_sclk) begin
		spi_miso <= buf_miso;
	end

	always @(posedge spi_sclk) begin
		buf_mosi <= spi_mosi;
	end


	// --------------------------
	// the spi transceiver

	reg [2:0] state;
	reg [7:0] word0, word1;

	assign buf_miso = word0[7];

	reg [4:0] queue_so;
	reg first_so;
	reg first_si;

	reg [2:0] seq_sclk;
	reg [2:0] bitcount;

	always @(posedge clk) begin
		seq_sclk <= {spi_sclk, seq_sclk[2:1]};
		queue_so <= queue_so[4:1];

		spi_ctrl_si <= 0;
		spi_ctrl_so <= 0;
		spi_ctrl_hd <= 0;

		if (!cselected) begin
			queue_so <= 'b10100;
			first_si <= 1;
			first_so <= 1;
			bitcount <= 0;
		end else begin
			if (queue_so[2]) begin
				spi_ctrl_so <= 1;
				spi_ctrl_hd <= first_so;
				first_so <= 0;
			end
			if (queue_so[0]) begin
				word1 <= spi_ctrl_do;
				word0 <= word1;
			end
			if (seq_sclk[1:0] == 2'b10) begin
				if (&bitcount) begin
					spi_ctrl_di <= {word0[6:0], buf_mosi};
					spi_ctrl_hd <= first_si;
					spi_ctrl_si <= 1;
					spi_ctrl_so <= 1;
					queue_so[1] <= 1;
					first_si <= 0;
				end else begin
					word0 <= {word0[6:0], buf_mosi};
				end
				bitcount <= bitcount + 1;
			end
		end
	end
endmodule
