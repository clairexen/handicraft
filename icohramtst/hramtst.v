module top (
	input CLKIN,
	output reg LED1, LED2, LED3,

	// RasPi Interface: 9 Data Lines (cmds have MSB set)
	inout RASPI_11, RASPI_12, RASPI_15, RASPI_16, RASPI_19, RASPI_21, RASPI_26, RASPI_35, RASPI_36,

	// RasPi Interface: Control Lines
	input RASPI_38, RASPI_40,

	// HRAM Interface
	output HRAM_CK, HRAM_CS,
	inout HRAM_RWDS, HRAM_DQ0, HRAM_DQ1, HRAM_DQ2, HRAM_DQ3, HRAM_DQ4, HRAM_DQ5, HRAM_DQ6, HRAM_DQ7
);
	wire clk, clk90, resetn;

	assign LED1 = 1;
	assign LED2 = 1;
	assign LED3 = 1;

	// ---- HRAM IO Pins ----

	reg hram_ck, hram_cs;

	reg hram_rwds_dir;
	reg hram_rwds_dout;
	wire hram_rwds_din;

	reg hram_dq_dir;
	reg [7:0] hram_dq_dout;
	wire [7:0] hram_dq_din;

	assign HRAM_CK = hram_ck;
	assign HRAM_CS = hram_cs;

	SB_IO #(
		.PIN_TYPE(6'b 1010_01),
		.PULLUP(1'b 0)
	) hram_rwds_io (
		.PACKAGE_PIN(HRAM_RWDS),
		.OUTPUT_ENABLE(hram_rwds_dir),
		.D_OUT_0(hram_rwds_dout),
		.D_IN_0(hram_rwds_din)
	);

	SB_IO #(
		.PIN_TYPE(6'b 1010_01),
		.PULLUP(1'b 0)
	) hram_dq_io [7:0] (
		.PACKAGE_PIN({HRAM_DQ7, HRAM_DQ6, HRAM_DQ5, HRAM_DQ4, HRAM_DQ3, HRAM_DQ2, HRAM_DQ1, HRAM_DQ0}),
		.OUTPUT_ENABLE(hram_dq_dir),
		.D_OUT_0(hram_dq_dout),
		.D_IN_0(hram_dq_din)
	);

	reg [15:0] libretto [0:1023];
	reg [15:0] pc;

	initial $readmemh("libretto_seed.hex", libretto);

	always @(posedge clk) begin
		if (!resetn) begin
			pc <= 0;
			hram_ck <= 1;
			hram_cs <= 1;
			hram_rwds_dir <= 0;
			hram_rwds_dout <= 0;
			hram_dq_dir <= 0;
			hram_dq_dout <= 0;
		end else begin
			{hram_ck, hram_cs, hram_rwds_dir, hram_rwds_dout, hram_dq_dir, hram_dq_dout} <= libretto[pc];
			pc <= pc + 1;
		end
	end

	// ---- Clock Generator ----

	wire clk_100mhz = CLKIN;
	wire pll_locked;

	SB_PLL40_2F_CORE #(
		.FEEDBACK_PATH("PHASE_AND_DELAY"),
		.DELAY_ADJUSTMENT_MODE_FEEDBACK("FIXED"),
		.DELAY_ADJUSTMENT_MODE_RELATIVE("FIXED"),
		.PLLOUT_SELECT_PORTA("SHIFTREG_0deg"),
		.PLLOUT_SELECT_PORTB("SHIFTREG_90deg"),
		.SHIFTREG_DIV_MODE(1'b0),
		.FDA_FEEDBACK(4'b1111),
		.FDA_RELATIVE(4'b1111),
		.DIVR(4'b0100),
		.DIVF(7'b0000000),
		.DIVQ(3'b101),
		.FILTER_RANGE(3'b111)
	) pll2 (
		.REFERENCECLK(clk_100mhz),
		.PLLOUTGLOBALA(clk),
		.PLLOUTGLOBALB(clk90),
		.LOCK(pll_locked),
		.BYPASS(1'b0),
		.RESETB(1'b1)
	);

	// ---- Reset Generator ----

	reg [7:0] resetn_counter = 0;
	assign resetn = &resetn_counter;

	always @(posedge clk) begin
		if (!pll_locked)
			resetn_counter <= 0;
		else if (!resetn)
			resetn_counter <= resetn_counter + 1;
	end

	// ---- Raspi Interface ----

	wire recv_sync;

	// recv ep0: transmission test
	wire recv_ep0_valid;
	wire recv_ep0_ready;
	wire [7:0] recv_ep0_data;

	// recv ep1: unused
	wire recv_ep1_valid;
	wire recv_ep1_ready = 1;
	wire [7:0] recv_ep1_data = recv_ep0_data;

	// recv ep2: unused
	wire recv_ep2_valid;
	wire recv_ep2_ready = 1;
	wire [7:0] recv_ep2_data = recv_ep0_data;

	// recv ep3: unused
	wire recv_ep3_valid;
	wire recv_ep3_ready = 1;
	wire [7:0] recv_ep3_data = recv_ep0_data;

	// send ep0: transmission test
	wire send_ep0_valid;
	wire send_ep0_ready;
	wire [7:0] send_ep0_data;

	// send ep1: debugger
	wire send_ep1_valid;
	wire send_ep1_ready;
	wire [7:0] send_ep1_data;

	// send ep2: unused
	wire send_ep2_valid = 0;
	wire send_ep2_ready;
	reg [7:0] send_ep2_data = 'bx;

	// send ep3: unused
	wire send_ep3_valid = 0;
	wire send_ep3_ready;
	wire [7:0] send_ep3_data = 'bx;

	// trigger lines
	wire trigger_0;	 // unused
	wire trigger_1;	 // debugger
	wire trigger_2;	 // unused
	wire trigger_3;	 // unused

	icosoc_raspif #(
		.NUM_RECV_EP(4),
		.NUM_SEND_EP(4),
		.NUM_TRIGGERS(4)
	) raspi_interface (
		.clk(clk),
		.sync(recv_sync),

		.recv_valid({
			recv_ep3_valid,
			recv_ep2_valid,
			recv_ep1_valid,
			recv_ep0_valid
		}),
		.recv_ready({
			recv_ep3_ready,
			recv_ep2_ready,
			recv_ep1_ready,
			recv_ep0_ready
		}),
		.recv_tdata(
			recv_ep0_data
		),

		.send_valid({
			send_ep3_valid,
			send_ep2_valid,
			send_ep1_valid,
			send_ep0_valid
		}),
		.send_ready({
			send_ep3_ready,
			send_ep2_ready,
			send_ep1_ready,
			send_ep0_ready
		}),
		.send_tdata(
			(send_ep3_data & {8{send_ep3_valid && send_ep3_ready}}) |
			(send_ep2_data & {8{send_ep2_valid && send_ep2_ready}}) |
			(send_ep1_data & {8{send_ep1_valid && send_ep1_ready}}) |
			(send_ep0_data & {8{send_ep0_valid && send_ep0_ready}})
		),

		.trigger({
			trigger_3,
			trigger_2,
			trigger_1,
			trigger_0
		}),

		.RASPI_11(RASPI_11),
		.RASPI_12(RASPI_12),
		.RASPI_15(RASPI_15),
		.RASPI_16(RASPI_16),
		.RASPI_19(RASPI_19),
		.RASPI_21(RASPI_21),
		.RASPI_26(RASPI_26),
		.RASPI_35(RASPI_35),
		.RASPI_36(RASPI_36),
		.RASPI_38(RASPI_38),
		.RASPI_40(RASPI_40)
	);

	// Transmission test (recv ep0, send ep0)
	assign send_ep0_data = ((recv_ep0_data << 5) + recv_ep0_data) ^ 7;
	assign send_ep0_valid = recv_ep0_valid;
	assign recv_ep0_ready = send_ep0_ready;

	// ---- Debugger ----

	wire debug_enable;
	wire debug_trigger;
	wire debug_triggered;
	wire [12:0] debug_data;

	icosoc_debugger #(
		.WIDTH(13),
		.DEPTH(1024),
		.TRIGAT(0),
		.MODE("FIRST_TRIGGER")
	) debugger (
		.clk(clk),
		.resetn(resetn),
		.enable(debug_enable),
		.trigger(debug_trigger),
		.triggered(debug_triggered),
		.data(debug_data),
		.dump_en(trigger_1),
		.dump_valid(send_ep1_valid),
		.dump_ready(send_ep1_ready),
		.dump_data(send_ep1_data)
	);

	assign debug_enable = 1;
	assign debug_trigger = 1;

	assign debug_data = {
		hram_ck,             // debug_12 -> hram_ck
		hram_cs,             // debug_11 -> hram_cs
		hram_rwds_dir,       // debug_10 -> hram_rwds_dir
		hram_rwds_din,       // debug_9  -> hram_rwds_din
		hram_dq_dir,         // debug_8  -> hram_dq_dir
		hram_dq_din[7],      // debug_7  -> hram_dq_din<7>
		hram_dq_din[6],      // debug_6  -> hram_dq_din<6>
		hram_dq_din[5],      // debug_5  -> hram_dq_din<5>
		hram_dq_din[4],      // debug_4  -> hram_dq_din<4>
		hram_dq_din[3],      // debug_3  -> hram_dq_din<3>
		hram_dq_din[2],      // debug_2  -> hram_dq_din<2>
		hram_dq_din[1],      // debug_1  -> hram_dq_din<1>
		hram_dq_din[0]       // debug_0  -> hram_dq_din<0>
	};
endmodule
