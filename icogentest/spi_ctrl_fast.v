`timescale 1 ns / 1 ps

module spi_ctrl_fast #(
	parameter NUM_ENDPOINTS = 3
) (
	input clk,
	input resetn,

	input       spi_sclk,
	input       spi_mosi,
	output      spi_miso,
	input [7:0] spi_csel,

	output reg spi_ctrl_si,
	output reg spi_ctrl_so,
	output reg spi_ctrl_hd,
	output reg [7:0] spi_ctrl_di,
	input  [7:0] spi_ctrl_do,

	output reg [NUM_ENDPOINTS-1:0] epsel
);
	reg [7:0] spi_csel_mask;
	wire [7:0] spi_csel_masked = spi_csel & spi_csel_mask;

	integer i;

	initial begin
		spi_csel_mask = 0;
		for (i = 0; i < NUM_ENDPOINTS; i = i+1)
			spi_csel_mask = spi_csel_mask | (i+1);
	end

	always @(posedge clk) begin
		epsel <= 0;
		for (i = 0; i < NUM_ENDPOINTS; i = i+1)
			if (spi_csel_masked == i+1) epsel[i] <= 1;
	end

	// all one bit codes are gray codes..
	reg spi_to_di = 0;
	reg spi_to_di_q = 0;
	reg spi_to_di_qq = 0;
	reg di_from_spi = 0;

	reg [7:0] in_shift_reg, out_shift_reg;
	reg [2:0] shift_bit_cnt;

	reg spi_csel_armed = 0;
	reg spi_csel_masked_q = 0;
	reg spi_csel_masked_qq = 0;
	reg spi_ctrl_so_first = 0;
	reg spi_ctrl_so_second = 0;
	reg spi_ctrl_si_first = 0;

	reg new_xfer_sclk = 0;
	reg new_xfer_clk = 0;

	reg [7:0] di_buffer;

	reg do_buffer_0_direct;
	reg do_buffer_1_direct;
	reg [7:0] do_buffer_0, do_buffer_1;
	wire [7:0] do_buffer = do_buf_sel ?
			(do_buffer_1_direct ? spi_ctrl_do : do_buffer_1) :
			(do_buffer_0_direct ? spi_ctrl_do : do_buffer_0);
	reg do_buf_sel = 0, do_buf_active;

`ifndef SYNTHESIS
	event sample_do, sample_di;
  `define trig_sample_do -> sample_do;
  `define trig_sample_di -> sample_di;
`else
  `define trig_sample_do begin end
  `define trig_sample_di begin end
`endif

	assign spi_miso = out_shift_reg[7];

	always @(posedge spi_sclk) begin
		in_shift_reg <= {in_shift_reg[6:0], spi_mosi};

		if (new_xfer_clk != new_xfer_sclk) begin
			spi_to_di <= !spi_to_di;
			new_xfer_sclk <= new_xfer_clk;
		end else if (&shift_bit_cnt) begin
			spi_to_di <= !spi_to_di;
			di_buffer <= {in_shift_reg[6:0], spi_mosi};
		end
	end

	always @(negedge spi_sclk) begin
		shift_bit_cnt <= shift_bit_cnt + 1;
		out_shift_reg <= out_shift_reg << 1;

		if (new_xfer_clk != new_xfer_sclk) begin
			`trig_sample_do
			shift_bit_cnt <= 0;
			out_shift_reg <= do_buffer;
			do_buf_sel <= !do_buf_sel;
		end else if (&shift_bit_cnt) begin
			`trig_sample_do
			out_shift_reg <= do_buffer;
			do_buf_sel <= !do_buf_sel;
		end
	end

	always @(negedge clk) begin
		spi_to_di_qq <= spi_to_di_q;
		spi_csel_masked_qq <= spi_csel_masked_q;
	end

	always @(posedge clk) begin
		spi_to_di_q <= spi_to_di;

		spi_ctrl_so_first <= 0;
		spi_csel_masked_q <= |spi_csel_masked;
		if (spi_csel_masked_q == spi_csel_masked_qq) begin
			spi_ctrl_so_first <= spi_csel_armed && spi_csel_masked_qq;
			spi_csel_armed <= !spi_csel_masked_qq;
		end

		if (spi_ctrl_so_first)
			spi_ctrl_so_second <= 1;

		spi_ctrl_si <= 0;
		spi_ctrl_so <= spi_ctrl_so_first;
		spi_ctrl_hd <= spi_ctrl_so_first;
		if (spi_to_di_qq != di_from_spi) begin
			if (!spi_ctrl_so_second) `trig_sample_di
			spi_ctrl_si <= !spi_ctrl_so_second;
			spi_ctrl_so <= 1;
			spi_ctrl_hd <= spi_ctrl_si_first;
			spi_ctrl_di <= di_buffer;
			di_from_spi <= !di_from_spi;
			spi_ctrl_so_second <= 0;
			spi_ctrl_si_first <= spi_ctrl_so_second;
		end
	end

	always @(posedge clk) begin
		do_buffer_0_direct <= 0;
		do_buffer_1_direct <= 0;

		if (!spi_csel_masked_q && !spi_csel_masked_qq) begin
			do_buf_active <= do_buf_sel;
			new_xfer_clk <= !new_xfer_sclk;
		end

		if (spi_ctrl_so) begin
			do_buffer_1_direct <= do_buf_active;
			do_buffer_0_direct <= !do_buf_active;
			do_buf_active <= !do_buf_active;
		end

		if (do_buffer_0_direct)
			do_buffer_0 <= spi_ctrl_do;

		if (do_buffer_1_direct)
			do_buffer_1 <= spi_ctrl_do;
	end
endmodule
