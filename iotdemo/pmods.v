module pmod_none (
	input clk,
	input resetn,

	input ctrl_wr,
	input ctrl_rd,
	input [ 7:0] ctrl_addr,
	input [31:0] ctrl_wdat,
	output reg [31:0] ctrl_rdat,
	output reg ctrl_done,

	inout PMOD_1, PMOD_2, PMOD_3, PMOD_4,
	inout PMOD_7, PMOD_8, PMOD_9, PMOD_10,

	inout PMOD2_1, PMOD2_2, PMOD2_3, PMOD2_4,
	inout PMOD2_7, PMOD2_8, PMOD2_9, PMOD2_10
);
	always @(posedge clk) begin
		ctrl_rdat <= 'bx;
		ctrl_done <= (ctrl_wr || ctrl_rd) && !ctrl_done;
	end
endmodule

// =======================================================================

module pmod_gpio (
	input clk,
	input resetn,

	input ctrl_wr,
	input ctrl_rd,
	input [ 7:0] ctrl_addr,
	input [31:0] ctrl_wdat,
	output reg [31:0] ctrl_rdat,
	output reg ctrl_done,

	inout PMOD_1, PMOD_2, PMOD_3, PMOD_4,
	inout PMOD_7, PMOD_8, PMOD_9, PMOD_10,

	inout PMOD2_1, PMOD2_2, PMOD2_3, PMOD2_4,
	inout PMOD2_7, PMOD2_8, PMOD2_9, PMOD2_10
);
	reg [7:0] direction_reg;
	reg [7:0] data_out_reg;
	wire [7:0] data_in;

	SB_IO #(
		.PIN_TYPE(6'b 1010_01),
		.PULLUP(1'b 0)
	) pmod_io [7:0] (
		.PACKAGE_PIN({PMOD_10, PMOD_9, PMOD_8, PMOD_7, PMOD_4, PMOD_3, PMOD_2, PMOD_1}),
		.OUTPUT_ENABLE(direction_reg),
		.D_OUT_0(data_out_reg),
		.D_IN_0(data_in)
	);

	always @(posedge clk) begin
		ctrl_rdat <= 'bx;
		ctrl_done <= 0;
		if (!resetn) begin
			direction_reg <= 0;
			data_out_reg <= 0;
		end else
		if (!ctrl_done) begin
			if (ctrl_wr) begin
				if (ctrl_addr == 0) direction_reg <= ctrl_wdat;
				if (ctrl_addr == 4) data_out_reg <= ctrl_wdat;
				ctrl_done <= 1;
			end
			if (ctrl_rd) begin
				if (ctrl_addr == 0) ctrl_rdat <= direction_reg;
				if (ctrl_addr == 4) ctrl_rdat <= data_out_reg;
				if (ctrl_addr == 8) ctrl_rdat <= data_in;
				ctrl_done <= 1;
			end
		end
	end
endmodule

// =======================================================================

module pmod_spi (
	input clk,
	input resetn,

	input ctrl_wr,
	input ctrl_rd,
	input [ 7:0] ctrl_addr,
	input [31:0] ctrl_wdat,
	output reg [31:0] ctrl_rdat,
	output reg ctrl_done,

	inout PMOD_1, PMOD_2, PMOD_3, PMOD_4,
	inout PMOD_7, PMOD_8, PMOD_9, PMOD_10,

	inout PMOD2_1, PMOD2_2, PMOD2_3, PMOD2_4,
	inout PMOD2_7, PMOD2_8, PMOD2_9, PMOD2_10
);
	wire spi_miso;
	reg spi_mosi, spi_sclk, spi_cs;

	reg [7:0] prescale_cnt;
	reg [7:0] prescale_cfg;
	reg [7:0] spi_data;
	reg [3:0] spi_state;

	SB_IO #(
		.PIN_TYPE(6'b 0000_01),
		.PULLUP(1'b 0)
	) pmod_unused [3:0] (
		.PACKAGE_PIN({PMOD_10, PMOD_9, PMOD_4, PMOD_1})
	);

	SB_IO #(
		.PIN_TYPE(6'b 0000_01),
		.PULLUP(1'b 0)
	) pmod_miso (
		.PACKAGE_PIN(PMOD_3),
		.D_IN_0(spi_miso)
	);

	SB_IO #(
		.PIN_TYPE(6'b 0110_01),
		.PULLUP(1'b 0)
	) pmod_mosi (
		.PACKAGE_PIN(PMOD_2),
		.D_OUT_0(spi_mosi)
	);

	SB_IO #(
		.PIN_TYPE(6'b 0110_01),
		.PULLUP(1'b 0)
	) pmod_sclk (
		.PACKAGE_PIN(PMOD_8),
		.D_OUT_0(spi_sclk)
	);

	SB_IO #(
		.PIN_TYPE(6'b 0110_01),
		.PULLUP(1'b 0)
	) pmod_cs (
		.PACKAGE_PIN(PMOD_7),
		.D_OUT_0(spi_cs)
	);

	always @(posedge clk) begin
		ctrl_rdat <= 'bx;
		ctrl_done <= 0;
		if (!resetn) begin
			spi_mosi <= 0;
			spi_sclk <= 1;
			spi_cs <= 1;

			prescale_cnt <= 0;
			prescale_cfg <= 0;
			spi_state <= 0;
		end else
		if (!ctrl_done) begin
			if (ctrl_wr) begin
				ctrl_done <= 1;
				if (ctrl_addr == 0) prescale_cfg <= ctrl_wdat;
				if (ctrl_addr == 4) spi_cs <= ctrl_wdat;
				if (ctrl_addr == 8) begin
					if (!prescale_cnt) begin
						if (spi_state == 0) begin
							spi_data <= ctrl_wdat;
							spi_mosi <= ctrl_wdat[7];
						end else begin
							if (spi_state[0])
								spi_data <= {spi_data, spi_miso};
							else
								spi_mosi <= spi_data[7];
						end
					end
					spi_sclk <= spi_state[0];
					ctrl_done <= spi_state == 15 && prescale_cnt == prescale_cfg;
					spi_state <= spi_state + (prescale_cnt == prescale_cfg);
					prescale_cnt <= prescale_cnt == prescale_cfg ? 0 : prescale_cnt + 1;
				end
			end
			if (ctrl_rd) begin
				ctrl_done <= 1;
				if (ctrl_addr == 0) ctrl_rdat <= prescale_cfg;
				if (ctrl_addr == 4) ctrl_rdat <= spi_cs;
				if (ctrl_addr == 8) ctrl_rdat <= spi_data;
			end
		end
	end
endmodule

// =======================================================================

module pmod_rs232 (
	input clk,
	input resetn,

	input ctrl_wr,
	input ctrl_rd,
	input [ 7:0] ctrl_addr,
	input [31:0] ctrl_wdat,
	output reg [31:0] ctrl_rdat,
	output reg ctrl_done,

	inout PMOD_1, PMOD_2, PMOD_3, PMOD_4,
	inout PMOD_7, PMOD_8, PMOD_9, PMOD_10,

	inout PMOD2_1, PMOD2_2, PMOD2_3, PMOD2_4,
	inout PMOD2_7, PMOD2_8, PMOD2_9, PMOD2_10
);
	parameter integer BAUD_RATE = 9600;
	parameter integer CLOCK_FREQ_HZ = 6000000;
	localparam integer HALF_PERIOD = CLOCK_FREQ_HZ / (2 * BAUD_RATE);

	reg [7:0] send_din;
	wire [7:0] send_dout;
	reg send_shift_in;
	reg send_shift_out;
	wire [7:0] send_used_slots;
	wire [7:0] send_free_slots;

	reg [7:0] recv_din;
	wire [7:0] recv_dout;
	reg recv_shift_in;
	reg recv_shift_out;
	wire [7:0] recv_used_slots;
	wire [7:0] recv_free_slots;

	wire pmod_rxd;
	reg pmod_txd;

	SB_IO #(
		.PIN_TYPE(6'b 0000_01),
		.PULLUP(1'b 0)
	) pmod_unused_io [5:0] (
		.PACKAGE_PIN({PMOD_10, PMOD_9, PMOD_8, PMOD_7, PMOD_4, PMOD_1})
	);

	SB_IO #(
		.PIN_TYPE(6'b 0000_01),
		.PULLUP(1'b 0)
	) pmod_rxd_io (
		.PACKAGE_PIN(PMOD_3),
		.D_IN_0(pmod_rxd)
	);

	SB_IO #(
		.PIN_TYPE(6'b 0110_01),
		.PULLUP(1'b 0)
	) pmod_txd_io (
		.PACKAGE_PIN(PMOD_2),
		.D_OUT_0(pmod_txd)
	);

	reg [$clog2(3*HALF_PERIOD):0] rx_cnt;
	reg [3:0] rx_state;
	reg pmod_rxd_q;

	always @(posedge clk) begin
		pmod_rxd_q <= pmod_rxd;
		recv_shift_in <= 0;

		if (!resetn) begin
			rx_state <= 0;
			rx_cnt <= 0;
		end else
		if (rx_cnt) begin
			rx_cnt <= rx_cnt - |1;
		end else
		if (rx_state == 0) begin
			if (pmod_rxd_q && !pmod_rxd) begin
				rx_state <= rx_state + |1;
				rx_cnt <= 3*HALF_PERIOD;
			end
		end else begin
			recv_din <= {pmod_rxd, recv_din[7:1]};
			rx_state <= rx_state + |1;
			rx_cnt <= 2*HALF_PERIOD;

			if (rx_state == 8) begin
				recv_shift_in <= 1;
				rx_state <= 0;
			end
		end
	end

	reg [$clog2(2*HALF_PERIOD):0] tx_cnt;
	reg [3:0] tx_state;
	reg [7:0] tx_byte;

	always @(posedge clk) begin
		send_shift_out <= 0;
		if (!resetn) begin
			pmod_txd <= 1;
			tx_state <= 0;
			tx_cnt <= 0;
		end else
		if (tx_cnt) begin
			tx_cnt <= tx_cnt - |1;
		end else
		if (tx_state == 0) begin
			if (|send_used_slots) begin
				pmod_txd <= 0;
				send_shift_out <= 1;
				tx_byte <= send_dout;
				tx_cnt <= 2*HALF_PERIOD;
				tx_state <= 1;
			end
		end else begin
			pmod_txd <= tx_byte[0];
			tx_byte <= tx_byte[7:1];
			tx_cnt <= 2*HALF_PERIOD;
			tx_state <= tx_state + |1;

			if (tx_state == 9) begin
				pmod_txd <= 1;
				tx_state <= 0;
			end
		end
	end

	pmod_rs232_fifo send_fifo (
		.clk       (clk            ),
		.resetn    (resetn         ),
		.din       (send_din       ),
		.dout      (send_dout      ),
		.shift_in  (send_shift_in  ),
		.shift_out (send_shift_out ),
		.used_slots(send_used_slots),
		.free_slots(send_free_slots)
	);

	pmod_rs232_fifo recv_fifo (
		.clk       (clk            ),
		.resetn    (resetn         ),
		.din       (recv_din       ),
		.dout      (recv_dout      ),
		.shift_in  (recv_shift_in  ),
		.shift_out (recv_shift_out ),
		.used_slots(recv_used_slots),
		.free_slots(recv_free_slots)
	);

	always @(posedge clk) begin
		ctrl_rdat <= 'bx;
		ctrl_done <= 0;

		recv_shift_out <= 0;
		send_shift_in <= 0;
		send_din <= 'bx;

		// Register file:
		//   0x00 shift data to/from send/recv fifos
		//   0x04 number of unread bytes in recv fifo (read-only)
		//   0x08 number of free bytes in send fifo (read-only)
		if (resetn && !ctrl_done) begin
			if (ctrl_wr) begin
				if (ctrl_addr == 0) begin
					send_shift_in <= 1;
					send_din <= ctrl_wdat;
				end
				ctrl_done <= 1;
			end
			if (ctrl_rd) begin
				if (ctrl_addr == 0) begin
					recv_shift_out <= 1;
					ctrl_rdat <= recv_dout;
				end
				if (ctrl_addr == 4) ctrl_rdat <= recv_used_slots;
				if (ctrl_addr == 8) ctrl_rdat <= send_free_slots;
				ctrl_done <= 1;
			end
		end
	end
endmodule

module pmod_rs232_fifo (
	input clk,
	input resetn,

	input [7:0] din,
	output [7:0] dout,

	input shift_in,
	input shift_out,
	output reg [7:0] used_slots,
	output reg [7:0] free_slots
);
	reg [7:0] memory [0:255];
	reg [7:0] wptr, rptr;

	reg [7:0] memory_dout;
	reg [7:0] pass_dout;
	reg use_pass_dout;

	assign dout = use_pass_dout ? pass_dout : memory_dout;

	wire do_shift_in = shift_in && |free_slots;
	wire do_shift_out = shift_out && |used_slots;

	always @(posedge clk) begin
		if (!resetn) begin
			wptr <= 0;
			rptr <= 0;
			used_slots <= 0;
			free_slots <= 255;
		end else begin
			memory[wptr] <= din;
			wptr <= wptr + do_shift_in;

			memory_dout <= memory[rptr + do_shift_out];
			rptr <= rptr + do_shift_out;

			use_pass_dout <= wptr == rptr;
			pass_dout <= din;

			if (do_shift_in && !do_shift_out) begin
				used_slots <= used_slots + 1;
				free_slots <= free_slots - 1;
			end

			if (!do_shift_in && do_shift_out) begin
				used_slots <= used_slots - 1;
				free_slots <= free_slots + 1;
			end
		end
	end
endmodule

// =======================================================================

module pmod_7segment (
	input clk,
	input resetn,

	input ctrl_wr,
	input ctrl_rd,
	input [ 7:0] ctrl_addr,
	input [31:0] ctrl_wdat,
	output reg [31:0] ctrl_rdat,
	output reg ctrl_done,

	inout PMOD_1, PMOD_2, PMOD_3, PMOD_4,
	inout PMOD_7, PMOD_8, PMOD_9, PMOD_10,

	inout PMOD2_1, PMOD2_2, PMOD2_3, PMOD2_4,
	inout PMOD2_7, PMOD2_8, PMOD2_9, PMOD2_10
);
	reg [3:0] digits [0:3];
	reg [3:0] decimal_points;
	reg [9:0] state = 0;
	reg [1:0] index = 0;

	reg [1:0] st1_index;
	reg [3:0] st1_digit;

	reg [3:0] st2_onehot;
	reg [7:0] st2_leds;

	always @(posedge clk) begin
		state <= state + 1;

		if (state == 8) begin
			index <= index + 1'b1;
		end

		st1_index <= index;
		st1_digit <= digits[index];

		st2_onehot <= 1 << st1_index;
		st2_leds <= decimal_points[st1_index] << 7;

		case (st1_digit)
			0: st2_leds[6:0] <= 7'b0111111;
			1: st2_leds[6:0] <= 7'b0000110;
			2: st2_leds[6:0] <= 7'b1011011;
			3: st2_leds[6:0] <= 7'b1001111;
			4: st2_leds[6:0] <= 7'b1100110;
			5: st2_leds[6:0] <= 7'b1101101;
			6: st2_leds[6:0] <= 7'b1111101;
			7: st2_leds[6:0] <= 7'b0000111;
			8: st2_leds[6:0] <= 7'b1111111;
			9: st2_leds[6:0] <= 7'b1101111;
		endcase

		if (state < 16)
			st2_onehot <= 0;
	end

	SB_IO #(
		.PIN_TYPE(6'b 0110_01),
		.PULLUP(1'b 0)
	) pmod_io [7:0] (
		.PACKAGE_PIN({PMOD_10, PMOD_9, PMOD_8, PMOD_7, PMOD_4, PMOD_3, PMOD_2, PMOD_1}),
		.D_OUT_0({!st2_leds[3], !st2_leds[7], !st2_leds[0], !st2_leds[6],
		          !st2_leds[4], !st2_leds[2], !st2_leds[1], !st2_leds[5]})
	);

	SB_IO #(
		.PIN_TYPE(6'b 0110_01),
		.PULLUP(1'b 0)
	) pmod2_io [7:0] (
		.PACKAGE_PIN({PMOD2_10, PMOD2_9, PMOD2_8, PMOD2_7, PMOD2_4, PMOD2_3, PMOD2_2, PMOD2_1}),
		.D_OUT_0({2'b0, !st2_onehot[0], !st2_onehot[2], 2'b0, !st2_onehot[1], !st2_onehot[3]})
	);

	always @(posedge clk) begin
		ctrl_rdat <= 'bx;
		ctrl_done <= 0;
		if (!resetn) begin
			digits[0] <= 0;
			digits[1] <= 0;
			digits[2] <= 0;
			digits[3] <= 0;
			decimal_points <= 0;
		end else
		if (!ctrl_done) begin
			if (ctrl_wr) begin
				if (ctrl_addr < 16) digits[ctrl_addr>>2] <= ctrl_wdat;
				if (ctrl_addr == 16) decimal_points <= ctrl_wdat;
				ctrl_done <= 1;
			end
			if (ctrl_rd) begin
				if (ctrl_addr < 16) ctrl_rdat <= digits[ctrl_addr>>2];
				if (ctrl_addr == 16) ctrl_rdat <= decimal_points;
				ctrl_done <= 1;
			end
		end
	end
endmodule
