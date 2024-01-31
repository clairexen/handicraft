// Synplify Pro is not comfortable inferring iCE40 4K brams for
// clock domain crossing FIFOs. Yosys does not have this issue.
// `define NO_BEHAVIORAL_FIFO_MODEL

// Synplify Pro and LSE both have troubles implementing memory[]
// with BRAMs. This is a hack for comparing the toolchains.
// `define MINI_BOOT_MEM

// Synplify Pro and LSE both have troubles implementing the frame
// buffer with BRAMs. This is a hack for comparing the toolchains.
// `define NO_LEDPANEL

// Divide the 12 MHz by this power of two: 0=12MHz, 1=6MHz, 2=3MHz, ...
`define POW2CLOCKDIV 1

module c3demo (
	input CLK12MHZ,
	output reg DEBUG0, DEBUG1, LED1, LED2, LED3,

	output reg SPI_FLASH_CS,
	output reg SPI_FLASH_SCLK,
	output reg SPI_FLASH_MOSI,
	input      SPI_FLASH_MISO,

	// 32x32 LED Panel
	output PANEL_R0, PANEL_G0, PANEL_B0, PANEL_R1, PANEL_G1, PANEL_B1,
	output PANEL_A, PANEL_B, PANEL_C, PANEL_D, PANEL_CLK, PANEL_STB, PANEL_OE,

	// RasPi Interface: 9 Data Lines (cmds have MSB set)
	inout RASPI_11, RASPI_12, RASPI_15, RASPI_16, RASPI_19, RASPI_21, RASPI_24, RASPI_35, RASPI_36,

	// RasPi Interface: Control Lines
	input RASPI_38, RASPI_40,

	// SRAM Interface
	output SRAM_A0, SRAM_A1, SRAM_A2, SRAM_A3, SRAM_A4, SRAM_A5, SRAM_A6, SRAM_A7,
	output SRAM_A8, SRAM_A9, SRAM_A10, SRAM_A11, SRAM_A12, SRAM_A13, SRAM_A14, SRAM_A15,
	inout SRAM_D0, SRAM_D1, SRAM_D2, SRAM_D3, SRAM_D4, SRAM_D5, SRAM_D6, SRAM_D7,
	inout SRAM_D8, SRAM_D9, SRAM_D10, SRAM_D11, SRAM_D12, SRAM_D13, SRAM_D14, SRAM_D15,
	output SRAM_CE, SRAM_WE, SRAM_OE, SRAM_LB, SRAM_UB,

	// PMODS
	inout PMOD1_1, PMOD1_2, PMOD1_3, PMOD1_4, PMOD1_7, PMOD1_8, PMOD1_9, PMOD1_10,
	inout PMOD2_1, PMOD2_2, PMOD2_3, PMOD2_4, PMOD2_7, PMOD2_8, PMOD2_9, PMOD2_10,
	inout PMOD3_1, PMOD3_2, PMOD3_3, PMOD3_4, PMOD3_7, PMOD3_8, PMOD3_9, PMOD3_10,
	inout PMOD4_1, PMOD4_2, PMOD4_3, PMOD4_4, PMOD4_7, PMOD4_8, PMOD4_9, PMOD4_10
);
	// 2048 32bit words = 8k bytes memory
	// 1024 32bit words = 4k bytes memory
	// 128 32bit words = 512 bytes memory
`ifndef MINI_BOOT_MEM
	localparam BOOT_MEM_SIZE = 1024;
`else
	localparam BOOT_MEM_SIZE = 4;
`endif

	// wire dgb0, dbg1;
	// always @* DEBUG0 = dbg0;
	// always @* DEBUG1 = dbg1;


	// -------------------------------
	// PLL

	wire clk, clk90;
	wire resetn;

	c3demo_clkgen clkgen (
		.CLK12MHZ(CLK12MHZ),
		.clk(clk),
		.clk90(clk90),
		.resetn(resetn)
	);


	// -------------------------------
	// PMODs

	reg [7:0] pmod1_dir, pmod1_dout;
	wire [7:0] pmod1_din;

	reg [7:0] pmod2_dir, pmod2_dout;
	wire [7:0] pmod2_din;

	reg [7:0] pmod3_dir, pmod3_dout;
	wire [7:0] pmod3_din;

	reg [7:0] pmod4_dir, pmod4_dout;
	wire [7:0] pmod4_din;

	SB_IO #(
		.PIN_TYPE(6'b 1010_01),
		.PULLUP(1'b 0)
	) pmod1_io [7:0] (
		.PACKAGE_PIN({PMOD1_10, PMOD1_9, PMOD1_8, PMOD1_7, PMOD1_4, PMOD1_3, PMOD1_2, PMOD1_1}),
		.OUTPUT_ENABLE(pmod1_dir),
		.D_OUT_0(pmod1_dout),
		.D_IN_0(pmod1_din)
	), pmod2_io [7:0] (
		.PACKAGE_PIN({PMOD2_10, PMOD2_9, PMOD2_8, PMOD2_7, PMOD2_4, PMOD2_3, PMOD2_2, PMOD2_1}),
		.OUTPUT_ENABLE(pmod2_dir),
		.D_OUT_0(pmod2_dout),
		.D_IN_0(pmod2_din)
	), pmod3_io [7:0] (
		.PACKAGE_PIN({PMOD3_10, PMOD3_9, PMOD3_8, PMOD3_7, PMOD3_4, PMOD3_3, PMOD3_2, PMOD3_1}),
		.OUTPUT_ENABLE(pmod3_dir),
		.D_OUT_0(pmod3_dout),
		.D_IN_0(pmod3_din)
	), pmod4_io [7:0] (
		.PACKAGE_PIN({PMOD4_10, PMOD4_9, PMOD4_8, PMOD4_7, PMOD4_4, PMOD4_3, PMOD4_2, PMOD4_1}),
		.OUTPUT_ENABLE(pmod4_dir),
		.D_OUT_0(pmod4_dout),
		.D_IN_0(pmod4_din)
	);


	// -------------------------------
	// SRAM Interface

	reg [1:0] sram_state;
	reg sram_wrlb, sram_wrub;
	reg [15:0] sram_addr, sram_dout;
	wire [15:0] sram_din;

`ifdef USE_TBUF
	assign sram_din = {SRAM_D15, SRAM_D14, SRAM_D13, SRAM_D12, SRAM_D11, SRAM_D10, SRAM_D9, SRAM_D8,
			SRAM_D7, SRAM_D6, SRAM_D5, SRAM_D4, SRAM_D3, SRAM_D2, SRAM_D1, SRAM_D0};

	assign {SRAM_D15, SRAM_D14, SRAM_D13, SRAM_D12, SRAM_D11, SRAM_D10, SRAM_D9, SRAM_D8,
			SRAM_D7, SRAM_D6, SRAM_D5, SRAM_D4, SRAM_D3, SRAM_D2, SRAM_D1, SRAM_D0} = (sram_wrlb || sram_wrub) ? sram_dout : 16'bz;
`else
	SB_IO #(
		.PIN_TYPE(6'b 1010_01),
		.PULLUP(1'b 0)
	) sram_io [15:0] (
		.PACKAGE_PIN({SRAM_D15, SRAM_D14, SRAM_D13, SRAM_D12, SRAM_D11, SRAM_D10, SRAM_D9, SRAM_D8,
		              SRAM_D7, SRAM_D6, SRAM_D5, SRAM_D4, SRAM_D3, SRAM_D2, SRAM_D1, SRAM_D0}),
		.OUTPUT_ENABLE(sram_wrlb || sram_wrub),
		.D_OUT_0(sram_dout),
		.D_IN_0(sram_din)
	);
`endif

	assign {SRAM_A15, SRAM_A14, SRAM_A13, SRAM_A12, SRAM_A11, SRAM_A10, SRAM_A9, SRAM_A8,
			SRAM_A7, SRAM_A6, SRAM_A5, SRAM_A4, SRAM_A3, SRAM_A2, SRAM_A1, SRAM_A0} = sram_addr;
	
	assign SRAM_CE = 0;
	assign SRAM_WE = (sram_wrlb || sram_wrub) ? !clk90 : 1;
	assign SRAM_OE = (sram_wrlb || sram_wrub);
	assign SRAM_LB = (sram_wrlb || sram_wrub) ? !sram_wrlb : 0;
	assign SRAM_UB = (sram_wrlb || sram_wrub) ? !sram_wrub : 0;


	// -------------------------------
	// RasPi Interface

	wire recv_sync;

	// recv ep0: transmission test
	wire recv_ep0_valid;
	wire recv_ep0_ready;
	wire [7:0] recv_ep0_data;

	// recv ep1: firmware upload
	wire recv_ep1_valid;
	wire recv_ep1_ready = 1;
	wire [7:0] recv_ep1_data = recv_ep0_data;

	// recv ep2: console input
	wire recv_ep2_valid;
	reg  recv_ep2_ready;
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

	// send ep2: console output
	reg  send_ep2_valid;
	wire send_ep2_ready;
	reg  [7:0] send_ep2_data;

	// send ep3: unused
	wire send_ep3_valid = 0;
	wire send_ep3_ready;
	wire [7:0] send_ep3_data = 'bx;

	// trigger lines
	wire trigger_0;  // debugger
	wire trigger_1;  // unused
	wire trigger_2;  // unused
	wire trigger_3;  // unused

	c3demo_raspi_interface #(
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
		.RASPI_24(RASPI_24),
		.RASPI_35(RASPI_35),
		.RASPI_36(RASPI_36),
		.RASPI_38(RASPI_38),
		.RASPI_40(RASPI_40)
	);


	// -------------------------------
	// Transmission test (recv ep0, send ep0) 

	assign send_ep0_data = ((recv_ep0_data << 5) + recv_ep0_data) ^ 7;
	assign send_ep0_valid = recv_ep0_valid;
	assign recv_ep0_ready = send_ep0_ready;
	

	// -------------------------------
	// Firmware upload (recv ep1)

	reg [15:0] prog_mem_addr;
	reg [31:0] prog_mem_data;
	reg [1:0] prog_mem_state;
	reg prog_mem_active = 0;
	reg prog_mem_reset = 0;

	always @(posedge clk) begin
		if (recv_sync) begin
			prog_mem_addr <= ~0;
			prog_mem_data <= 0;
			prog_mem_state <= 0;
			prog_mem_active <= 0;
			prog_mem_reset <= 0;
		end else
		if (recv_ep1_valid) begin
			prog_mem_addr <= prog_mem_addr + &prog_mem_state;
			prog_mem_data <= {recv_ep1_data, prog_mem_data[31:8]};
			prog_mem_state <= prog_mem_state + 1;
			prog_mem_active <= &prog_mem_state;
			prog_mem_reset <= 1;
		end
	end


	// -------------------------------
	// LED Panel Driver

	reg led_wr_enable;
	reg [4:0] led_wr_addr_x = 0;
	reg [4:0] led_wr_addr_y = 0;
	reg [23:0] led_wr_rgb_data;

`ifndef NO_LEDPANEL
	ledpanel ledpanel (
		.clk        (clk            ),
		.wr_enable  (led_wr_enable  ),
		.wr_addr_x  (led_wr_addr_x  ),
		.wr_addr_y  (led_wr_addr_y  ),
		.wr_rgb_data(led_wr_rgb_data),

		.PANEL_R0   (PANEL_R0 ),
		.PANEL_G0   (PANEL_G0 ),
		.PANEL_B0   (PANEL_B0 ),
		.PANEL_R1   (PANEL_R1 ),
		.PANEL_G1   (PANEL_G1 ),
		.PANEL_B1   (PANEL_B1 ),
		.PANEL_A    (PANEL_A  ),
		.PANEL_B    (PANEL_B  ),
		.PANEL_C    (PANEL_C  ),
		.PANEL_D    (PANEL_D  ),
		.PANEL_CLK  (PANEL_CLK),
		.PANEL_STB  (PANEL_STB),
		.PANEL_OE   (PANEL_OE )
	);
`endif


	// -------------------------------
	// PicoRV32 Core

	wire cpu_trap;
	wire mem_valid;
	wire mem_instr;
	wire [31:0] mem_addr;
	wire [31:0] mem_wdata;
	wire [3:0] mem_wstrb;

	reg mem_ready;
	reg [31:0] mem_rdata;

	wire resetn_picorv32 = resetn && !prog_mem_reset;

	picorv32 #(
		.ENABLE_IRQ(1)
	) cpu (
		.clk       (clk            ),
		.resetn    (resetn_picorv32),
		.trap      (cpu_trap       ),
		.mem_valid (mem_valid      ),
		.mem_instr (mem_instr      ),
		.mem_ready (mem_ready      ),
		.mem_addr  (mem_addr       ),
		.mem_wdata (mem_wdata      ),
		.mem_wstrb (mem_wstrb      ),
		.mem_rdata (mem_rdata      ),
		.irq       (32'b0          )
	);


	// -------------------------------
	// On-chip logic analyzer (send ep1, trig1)

	wire debug_enable;
	wire debug_trigger;
	wire debug_triggered;
	wire [30:0] debug_data;

	c3demo_debugger #(
		.WIDTH(31),
		.DEPTH(256),
		.TRIGAT(192),
		.MODE("FREE_RUNNING")
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
		cpu_trap,          // debug_30 -> cpu_trap
		mem_wstrb[3],      // debug_29 -> mem_wstrb_3
		mem_wstrb[2],      // debug_28 -> mem_wstrb_2
		mem_wstrb[1],      // debug_27 -> mem_wstrb_1
		mem_wstrb[0],      // debug_26 -> mem_wstrb_0
		mem_valid,         // debug_25 -> mem_valid
		mem_ready,         // debug_24 -> mem_ready
		mem_instr,         // debug_23 -> mem_instr
		mem_addr[31],      // debug_22 -> addr_31
		mem_addr[30],      // debug_21 -> addr_30
		mem_addr[29],      // debug_20 -> addr_29
		mem_addr[28],      // debug_19 -> addr_28
		|mem_addr[31:18],  // debug_18 -> addr_hi
		mem_addr[17],      // debug_17 -> addr_17
		mem_addr[16],      // debug_16 -> addr_16
		mem_addr[15],      // debug_15 -> addr_15
		mem_addr[14],      // debug_14 -> addr_14
		mem_addr[13],      // debug_13 -> addr_13
		mem_addr[12],      // debug_12 -> addr_12
		mem_addr[11],      // debug_11 -> addr_11
		mem_addr[10],      // debug_10 -> addr_10
		mem_addr[9],       // debug_9  -> addr_9
		mem_addr[8],       // debug_8  -> addr_8
		mem_addr[7],       // debug_7  -> addr_7
		mem_addr[6],       // debug_6  -> addr_6
		mem_addr[5],       // debug_5  -> addr_5
		mem_addr[4],       // debug_4  -> addr_4
		mem_addr[3],       // debug_3  -> addr_3
		mem_addr[2],       // debug_2  -> addr_2
		mem_addr[1],       // debug_1  -> addr_1
		mem_addr[0]        // debug_0  -> addr_0
	};


	// -------------------------------
	// Memory/IO Interface

	reg [31:0] memory [0:BOOT_MEM_SIZE-1];
	initial $readmemh("firmware.hex", memory);

	always @(posedge clk) begin
		mem_ready <= 0;
		led_wr_enable <= 0;

		sram_state <= 0;
		sram_wrlb <= 0;
		sram_wrub <= 0;
		sram_addr <= 'bx;
		sram_dout <= 'bx;

		if (send_ep2_ready)
			send_ep2_valid <= 0;

		recv_ep2_ready <= 0;

		if (!resetn_picorv32) begin
			LED1 <= 0;
			LED2 <= 0;
			LED3 <= 0;
			DEBUG0 <= 0;
			DEBUG1 <= 0;

			SPI_FLASH_CS   <= 1;
			SPI_FLASH_SCLK <= 1;
			SPI_FLASH_MOSI <= 0;

			send_ep2_valid <= 0;

			if (prog_mem_active) begin
`ifndef MINI_BOOT_MEM
				memory[prog_mem_addr] <= prog_mem_data;
`endif
			end
		end else
		if (mem_valid && !mem_ready) begin
			(* parallel_case *)
			case (1)
				(mem_addr >> 2) < BOOT_MEM_SIZE: begin
					if (mem_wstrb) begin
`ifndef MINI_BOOT_MEM
						if (mem_wstrb[0]) memory[mem_addr >> 2][ 7: 0] <= mem_wdata[ 7: 0];
						if (mem_wstrb[1]) memory[mem_addr >> 2][15: 8] <= mem_wdata[15: 8];
						if (mem_wstrb[2]) memory[mem_addr >> 2][23:16] <= mem_wdata[23:16];
						if (mem_wstrb[3]) memory[mem_addr >> 2][31:24] <= mem_wdata[31:24];
`endif
					end else begin
						mem_rdata <= memory[mem_addr >> 2];
					end
					mem_ready <= 1;
				end
				(mem_addr & 32'hF000_0000) == 32'h0000_0000 && (mem_addr >> 2) >= BOOT_MEM_SIZE: begin
					if (mem_wstrb) begin
						(* parallel_case, full_case *)
						case (sram_state)
							0: begin
								sram_addr <= {mem_addr >> 2, 1'b0};
								sram_dout <= mem_wdata[15:0];
								sram_wrlb <= mem_wstrb[0];
								sram_wrub <= mem_wstrb[1];
								sram_state <= 1;
							end
							1: begin
								sram_addr <= {mem_addr >> 2, 1'b1};
								sram_dout <= mem_wdata[31:16];
								sram_wrlb <= mem_wstrb[2];
								sram_wrub <= mem_wstrb[3];
								sram_state <= 0;
								mem_ready <= 1;
							end
						endcase
					end else begin
						(* parallel_case, full_case *)
						case (sram_state)
							0: begin
								sram_addr <= {mem_addr >> 2, 1'b0};
								sram_state <= 1;
							end
							1: begin
								sram_addr <= {mem_addr >> 2, 1'b1};
								mem_rdata[15:0] <= sram_din;
								sram_state <= 2;
							end
							2: begin
								mem_rdata[31:16] <= sram_din;
								sram_state <= 0;
								mem_ready <= 1;
							end
						endcase
					end
				end
				(mem_addr & 32'hF000_0000) == 32'h1000_0000: begin
					if (mem_wstrb) begin
						{led_wr_addr_y, led_wr_addr_x} <= mem_addr >> 2;
						led_wr_rgb_data <= mem_wdata;
						led_wr_enable <= 1;
					end
					mem_ready <= 1;
				end
				(mem_addr & 32'hF000_0000) == 32'h2000_0000: begin
					if (mem_wstrb) begin
						if (mem_addr[7:0] == 8'h 00) {DEBUG1, DEBUG0, LED3, LED2, LED1} <= mem_wdata;
						if (mem_addr[7:0] == 8'h 10) pmod1_dir <= mem_wdata;
						if (mem_addr[7:0] == 8'h 14) pmod1_dout <= mem_wdata;
						if (mem_addr[7:0] == 8'h 20) pmod2_dir <= mem_wdata;
						if (mem_addr[7:0] == 8'h 24) pmod2_dout <= mem_wdata;
						if (mem_addr[7:0] == 8'h 30) pmod3_dir <= mem_wdata;
						if (mem_addr[7:0] == 8'h 34) pmod3_dout <= mem_wdata;
						if (mem_addr[7:0] == 8'h 40) pmod4_dir <= mem_wdata;
						if (mem_addr[7:0] == 8'h 44) pmod4_dout <= mem_wdata;
						if (mem_addr[7:0] == 8'h 50) {SPI_FLASH_CS, SPI_FLASH_SCLK, SPI_FLASH_MOSI} <= mem_wdata[3:1];
					end
					mem_rdata <= 0;
					if (mem_addr[7:0] == 8'h 00) mem_rdata <= {DEBUG1, DEBUG0, LED3, LED2, LED1};
					if (mem_addr[7:0] == 8'h 14) mem_rdata <= pmod1_din;
					if (mem_addr[7:0] == 8'h 24) mem_rdata <= pmod2_din;
					if (mem_addr[7:0] == 8'h 34) mem_rdata <= pmod3_din;
					if (mem_addr[7:0] == 8'h 44) mem_rdata <= pmod4_din;
					if (mem_addr[7:0] == 8'h 50) mem_rdata <= {SPI_FLASH_CS, SPI_FLASH_SCLK, SPI_FLASH_MOSI, SPI_FLASH_MISO};
					mem_ready <= 1;
				end
				(mem_addr & 32'hF000_0000) == 32'h3000_0000: begin
					if (mem_wstrb) begin
						if (send_ep2_ready || !send_ep2_valid) begin
							send_ep2_valid <= 1;
							send_ep2_data <= mem_wdata;
							mem_ready <= 1;
						end
					end else begin
						if (recv_ep2_valid && !recv_ep2_ready) begin
							recv_ep2_ready <= 1;
							mem_rdata <= recv_ep2_data;
						end else begin
							mem_rdata <= ~0;
						end
						mem_ready <= 1;
					end
				end
			endcase
		end
	end
endmodule

// ======================================================================

module c3demo_clkgen (
	input CLK12MHZ,
	output clk, clk90,
	output resetn
);
	// PLLs are not working on alpha board 
	// -----------------------------------
	//
	// wire [7:0] DYNAMICDELAY = 0;
	// wire PLLOUTCORE, EXTFEEDBACK = 0, LATCHINPUTVALUE = 0;
	// SB_PLL40_CORE #(
	// 	.FEEDBACK_PATH("SIMPLE"),
	// 	.DELAY_ADJUSTMENT_MODE_FEEDBACK("FIXED"),
	// 	.DELAY_ADJUSTMENT_MODE_RELATIVE("FIXED"),
	// 	.PLLOUT_SELECT("GENCLK"),
	// 	.FDA_FEEDBACK(0),
	// 	.FDA_RELATIVE(0),
	// 	.DIVR(10),
	// 	.DIVF(0),
	// 	.DIVQ(1),
	// 	.FILTER_RANGE(0),
	// 	.ENABLE_ICEGATE(0),
	// 	.TEST_MODE(0)
	// ) uut (
	// 	.REFERENCECLK   (CLK12MHZ       ),
	// 	.PLLOUTGLOBAL   (clk            ),
	// 	.LOCK           (pll_lock       ),
	// 	.BYPASS         (1'b0           ),
	// 	.RESETB         (1'b1           ),
	// 	.PLLOUTCORE     (PLLOUTCORE     ),
	// 	.EXTFEEDBACK    (EXTFEEDBACK    ),
	// 	.DYNAMICDELAY   (DYNAMICDELAY   ),
	// 	.LATCHINPUTVALUE(LATCHINPUTVALUE)
	// );

	reg [`POW2CLOCKDIV:0] divided_clock = 'b0x;
	always @* divided_clock[0] = CLK12MHZ;

	genvar i;
	generate for (i = 1; i <= `POW2CLOCKDIV; i = i+1) begin
		always @(posedge divided_clock[i-1])
			divided_clock[i] <= !divided_clock[i];
	end endgenerate

	SB_GB clock_buffer (
		.USER_SIGNAL_TO_GLOBAL_BUFFER(divided_clock[`POW2CLOCKDIV]),
		.GLOBAL_BUFFER_OUTPUT(clk)
	);

	reg clk90_r;
	assign clk90 = clk90_r;

	always @(negedge divided_clock[`POW2CLOCKDIV-1])
		clk90_r <= clk;

	// -------------------------------
	// Reset Generator

	reg [7:0] resetn_counter = 0;
	assign resetn = &resetn_counter;

	always @(posedge clk) begin
		if (!resetn)
			resetn_counter <= resetn_counter + 1;
	end
endmodule

// ======================================================================

module c3demo_crossclkfifo #(
	parameter WIDTH = 8,
	parameter DEPTH = 16
) (
	input                  in_clk,
	input                  in_shift,
	input      [WIDTH-1:0] in_data,
	output reg             in_full,
	output reg             in_nempty,

	input                  out_clk,
	input                  out_pop,
	output     [WIDTH-1:0] out_data,
	output reg             out_nempty
);
	localparam integer ABITS = $clog2(DEPTH);

	initial begin
		in_full = 0;
		in_nempty = 0;
		out_nempty = 0;
	end

	function [ABITS-1:0] bin2gray(input [ABITS-1:0] in);
		begin
			bin2gray = in ^ (in >> 1);
		end
	endfunction

	function [ABITS-1:0] gray2bin(input [ABITS-1:0] in);
		begin
			gray2bin = in;
			// Assuming ABITS is less or equal 32
			gray2bin = gray2bin ^ (gray2bin >> 16);
			gray2bin = gray2bin ^ (gray2bin >> 8);
			gray2bin = gray2bin ^ (gray2bin >> 4);
			gray2bin = gray2bin ^ (gray2bin >> 2);
			gray2bin = gray2bin ^ (gray2bin >> 1);
		end
	endfunction

	reg [ABITS-1:0] in_ipos = 0, in_opos = 0;
	reg [ABITS-1:0] out_opos = 0, out_ipos = 0;


`ifndef NO_BEHAVIORAL_FIFO_MODEL

	// Behavioral model for the clock domain crossing fifo

	reg [WIDTH-1:0] memory [0:DEPTH-1];

	// input side of fifo

	wire [ABITS-1:0] next_ipos = in_ipos == DEPTH-1 ? 0 : in_ipos + 1;
	wire [ABITS-1:0] next_next_ipos = next_ipos == DEPTH-1 ? 0 : next_ipos + 1;

	always @(posedge in_clk) begin
		if (in_shift && !in_full) begin
			memory[in_ipos] <= in_data;
			in_full <= next_next_ipos == in_opos;
			in_nempty <= 1;
			in_ipos <= next_ipos;
		end else begin
			in_full <= next_ipos == in_opos;
			in_nempty <= in_ipos != in_opos;
		end
	end

	// output side of fifo

	wire [ABITS-1:0] next_opos = out_opos == DEPTH-1 ? 0 : out_opos + 1;
	reg [WIDTH-1:0] out_data_d;

	always @(posedge out_clk) begin
		if (out_pop && out_nempty) begin
			out_data_d <= memory[next_opos];
			out_nempty <= next_opos != out_ipos;
			out_opos <= next_opos;
		end else begin
			out_data_d <= memory[out_opos];
			out_nempty <= out_opos != out_ipos;
		end
	end

	assign out_data = out_nempty ? out_data_d : 0;

`else /* NO_BEHAVIORAL_FIFO_MODEL */

	// Structural model for the clock domain crossing fifo

	wire        memory_wclk  = in_clk;
	wire        memory_wclke = 1;
	wire        memory_we;
	wire [10:0] memory_waddr;
	wire [15:0] memory_mask  = 16'h 0000;
	wire [15:0] memory_wdata;

	wire [15:0] memory_rdata;
	wire        memory_rclk  = out_clk;
	wire        memory_rclke = 1;
	wire        memory_re    = 1;
	wire [10:0] memory_raddr;

	SB_RAM40_4K #(
		.WRITE_MODE(0),
		.READ_MODE(0)
	) memory (
		.WCLK (memory_wclk ),
		.WCLKE(memory_wclke),
		.WE   (memory_we   ),
		.WADDR(memory_waddr),
		.MASK (memory_mask ),
		.WDATA(memory_wdata),

		.RDATA(memory_rdata),
		.RCLK (memory_rclk ),
		.RCLKE(memory_rclke),
		.RE   (memory_re   ),
		.RADDR(memory_raddr)
	);

	initial begin
		if (WIDTH > 16 || DEPTH > 256) begin
			$display("Fifo with width %d and depth %d does not fit into a SB_RAM40_4K!", WIDTH, DEPTH);
			$finish;
		end
	end

	// input side of fifo

	wire [ABITS-1:0] next_ipos = in_ipos == DEPTH-1 ? 0 : in_ipos + 1;
	wire [ABITS-1:0] next_next_ipos = next_ipos == DEPTH-1 ? 0 : next_ipos + 1;

	always @(posedge in_clk) begin
		if (in_shift && !in_full) begin
			in_full <= next_next_ipos == in_opos;
			in_nempty <= 1;
			in_ipos <= next_ipos;
		end else begin
			in_full <= next_ipos == in_opos;
			in_nempty <= in_ipos != in_opos;
		end
	end

	assign memory_we = in_shift && !in_full;
	assign memory_waddr = in_ipos;
	assign memory_wdata = in_data;

	// output side of fifo

	wire [ABITS-1:0] next_opos = out_opos == DEPTH-1 ? 0 : out_opos + 1;
	wire [WIDTH-1:0] out_data_d = memory_rdata;

	always @(posedge out_clk) begin
		if (out_pop && out_nempty) begin
			out_nempty <= next_opos != out_ipos;
			out_opos <= next_opos;
		end else begin
			out_nempty <= out_opos != out_ipos;
		end
	end

	assign memory_raddr = (out_pop && out_nempty) ? next_opos : out_opos;
	assign out_data = out_nempty ? out_data_d : 0;

`endif /* NO_BEHAVIORAL_FIFO_MODEL */


	// clock domain crossing of ipos

	reg [ABITS-1:0] in_ipos_gray = 0;
	reg [ABITS-1:0] out_ipos_gray_2 = 0;
	reg [ABITS-1:0] out_ipos_gray_1 = 0;
	reg [ABITS-1:0] out_ipos_gray_0 = 0;

	always @(posedge in_clk) begin
		in_ipos_gray <= bin2gray(in_ipos);
	end

	always @(posedge out_clk) begin
		out_ipos_gray_2 <= in_ipos_gray;
		out_ipos_gray_1 <= out_ipos_gray_2;
		out_ipos_gray_0 <= out_ipos_gray_1;
		out_ipos <= gray2bin(out_ipos_gray_0);
	end


	// clock domain crossing of opos

	reg [ABITS-1:0] out_opos_gray = 0;
	reg [ABITS-1:0] in_opos_gray_2 = 0;
	reg [ABITS-1:0] in_opos_gray_1 = 0;
	reg [ABITS-1:0] in_opos_gray_0 = 0;

	always @(posedge out_clk) begin
		out_opos_gray <= bin2gray(out_opos);
	end

	always @(posedge in_clk) begin
		in_opos_gray_2 <= out_opos_gray;
		in_opos_gray_1 <= in_opos_gray_2;
		in_opos_gray_0 <= in_opos_gray_1;
		in_opos <= gray2bin(in_opos_gray_0);
	end
endmodule

// ======================================================================

module c3demo_raspi_interface #(
	// number of communication endpoints
	parameter NUM_RECV_EP = 4,
	parameter NUM_SEND_EP = 4,
	parameter NUM_TRIGGERS = 4
) (
	input clk,
	output sync,

	output [NUM_RECV_EP-1:0] recv_valid,
	input  [NUM_RECV_EP-1:0] recv_ready,
	output [       7:0] recv_tdata,

	input  [NUM_SEND_EP-1:0] send_valid,
	output [NUM_SEND_EP-1:0] send_ready,
	input  [       7:0] send_tdata,

	output [NUM_TRIGGERS-1:0] trigger,

	// RasPi Interface: 9 Data Lines (cmds have MSB set)
	inout RASPI_11, RASPI_12, RASPI_15, RASPI_16, RASPI_19, RASPI_21, RASPI_24, RASPI_35, RASPI_36,

	// RasPi Interface: Control Lines
	input RASPI_38, RASPI_40
);
	// All signals with "raspi_" prefix are in the "raspi_clk" clock domain.
	// All other signals are in the "clk" clock domain.

	wire [8:0] raspi_din;
	reg [8:0] raspi_dout;

	wire raspi_dir = RASPI_38;
	wire raspi_clk;

	SB_GB raspi_clock_buffer (
		.USER_SIGNAL_TO_GLOBAL_BUFFER(RASPI_40),
		.GLOBAL_BUFFER_OUTPUT(raspi_clk)
	);

	SB_IO #(
		.PIN_TYPE(6'b 1010_01),
		.PULLUP(1'b 0)
	) raspi_io [8:0] (
		.PACKAGE_PIN({RASPI_11, RASPI_12, RASPI_15, RASPI_16, RASPI_19, RASPI_21, RASPI_24, RASPI_35, RASPI_36}),
		.OUTPUT_ENABLE(!raspi_dir),
		.D_OUT_0(raspi_dout),
		.D_IN_0(raspi_din)
	);


	// system clock side

	function [NUM_SEND_EP-1:0] highest_send_bit;
		input [NUM_SEND_EP-1:0] bits;
		integer i;
		begin
			highest_send_bit = 0;
			for (i = 0; i < NUM_SEND_EP; i = i+1)
				if (bits[i]) highest_send_bit = 1 << i;
		end
	endfunction

	function [7:0] highest_send_bit_index;
		input [NUM_SEND_EP-1:0] bits;
		integer i;
		begin
			highest_send_bit_index = 0;
			for (i = 0; i < NUM_SEND_EP; i = i+1)
				if (bits[i]) highest_send_bit_index = i;
		end
	endfunction

	wire [7:0] recv_epnum, send_epnum;
	wire recv_nempty, send_full;

	assign recv_valid = recv_nempty ? 1 << recv_epnum : 0;
	assign send_ready = highest_send_bit(send_valid) & {NUM_SEND_EP{!send_full}};
	assign send_epnum = highest_send_bit_index(send_valid);

	assign sync = &recv_epnum && &recv_tdata && recv_nempty;
	assign trigger = &recv_epnum && recv_nempty ? 1 << recv_tdata : 0;


	// raspi side

	reg [7:0] raspi_din_ep;
	reg [7:0] raspi_dout_ep = 0;
	wire raspi_recv_nempty;

	wire [15:0] raspi_send_data;
	wire raspi_send_nempty;

	always @* begin
		raspi_dout = raspi_recv_nempty ? 9'h 1fe : 9'h 1ff;
		if (raspi_send_nempty) begin
			if (raspi_dout_ep != raspi_send_data[15:8])
				raspi_dout = {1'b1, raspi_send_data[15:8]};
			else
				raspi_dout = {1'b0, raspi_send_data[ 7:0]};
		end
	end

	always @(posedge raspi_clk) begin
		if (raspi_din[8] && raspi_dir)
			raspi_din_ep <= raspi_din[7:0];
		if (!raspi_dir)
			raspi_dout_ep <= raspi_send_nempty ? raspi_send_data[15:8] : raspi_dout;
	end


	// fifos

	c3demo_crossclkfifo #(
		.WIDTH(16),
		.DEPTH(256)
	) fifo_recv (
		.in_clk(raspi_clk),
		.in_shift(raspi_dir && !raspi_din[8]),
		.in_data({raspi_din_ep, raspi_din[7:0]}),
		.in_nempty(raspi_recv_nempty),

		.out_clk(clk),
		.out_pop(|(recv_valid & recv_ready) || (recv_epnum >= NUM_RECV_EP)),
		.out_data({recv_epnum, recv_tdata}),
		.out_nempty(recv_nempty)
	), fifo_send (
		.in_clk(clk),
		.in_shift(|(send_valid & send_ready)),
		.in_data({send_epnum, send_tdata}),
		.in_full(send_full),

		.out_clk(raspi_clk),
		.out_pop((raspi_dout_ep == raspi_send_data[15:8]) && !raspi_dir),
		.out_data(raspi_send_data),
		.out_nempty(raspi_send_nempty)
	);
endmodule

// ======================================================================

module c3demo_debugger #(
	parameter WIDTH = 32,
	parameter DEPTH = 256,
	parameter TRIGAT = 64,
	parameter MODE = "NORMAL"
) (
	input clk,
	input resetn,

	input             enable,
	input             trigger,
	output            triggered,
	input [WIDTH-1:0] data,

	input            dump_en,
	output reg       dump_valid,
	input            dump_ready,
	output reg [7:0] dump_data
);
	localparam DEPTH_BITS = $clog2(DEPTH);

	localparam BYTES = (WIDTH + 7) / 8;
	localparam BYTES_BITS = $clog2(BYTES);

	reg [WIDTH-1:0] memory [0:DEPTH-1];
	reg [DEPTH_BITS-1:0] mem_pointer, stop_counter;
	reg [BYTES_BITS-1:0] bytes_counter;
	reg dump_en_r;

	reg [1:0] state;
	localparam state_running   = 0;
	localparam state_triggered = 1;
	localparam state_waitdump  = 2;
	localparam state_dump      = 3;

	localparam mode_normal        = MODE == "NORMAL";         // first trigger after dump_en
	localparam mode_free_running  = MODE == "FREE_RUNNING";   // trigger on dump_en
	localparam mode_first_trigger = MODE == "FIRST_TRIGGER";  // block on first trigger
	localparam mode_last_trigger  = MODE == "LAST_TRIGGER";   // wait for last trigger

	initial begin
		if (!{mode_normal, mode_free_running, mode_first_trigger, mode_last_trigger}) begin
			$display("Invalid debugger MODE: %s", MODE);
			$finish;
		end
	end

	always @(posedge clk)
		dump_data <= memory[mem_pointer] >> (8*bytes_counter);
	
	assign triggered = resetn && state != state_running;

	always @(posedge clk) begin
		dump_valid <= 0;
		if (dump_en)
			dump_en_r <= 1;
		if (!resetn) begin
			mem_pointer <= 0;
			stop_counter <= DEPTH-1;
			state <= state_running;
			dump_en_r <= 0;
		end else
		case (state)
			state_running: begin
				if (enable) begin
					memory[mem_pointer] <= data;
					mem_pointer <= mem_pointer == DEPTH-1 ? 0 : mem_pointer+1;
					if (!stop_counter) begin
						if (mode_free_running) begin
							if (dump_en_r) begin
								state <= state_dump;
								stop_counter <= DEPTH-1;
								bytes_counter <= 0;
							end
						end else begin
							if (trigger && (dump_en_r || !mode_normal)) begin
								stop_counter <= DEPTH - TRIGAT - 2;
								state <= state_triggered;
							end
						end
					end else
						stop_counter <= stop_counter - 1;
				end
			end
			state_triggered: begin
				if (enable) begin
					memory[mem_pointer] <= data;
					mem_pointer <= mem_pointer == DEPTH-1 ? 0 : mem_pointer+1;
					stop_counter <= stop_counter - 1;
					if (mode_last_trigger && trigger) begin
						stop_counter <= DEPTH - TRIGAT - 2;
					end
					if (!stop_counter) begin
						state <= state_waitdump;
					end
				end
			end
			state_waitdump: begin
				if (dump_en_r)
					state <= state_dump;
				stop_counter <= DEPTH-1;
				bytes_counter <= 0;
			end
			state_dump: begin
				if (dump_valid && dump_ready) begin
					if (bytes_counter == BYTES-1) begin
						bytes_counter <= 0;
						stop_counter <= stop_counter - 1;
						mem_pointer <= mem_pointer == DEPTH-1 ? 0 : mem_pointer+1;
						if (!stop_counter) begin
							stop_counter <= DEPTH-1;
							state <= state_running;
							dump_en_r <= 0;
						end
					end else begin
						bytes_counter <= bytes_counter + 1;
					end
				end else begin
					dump_valid <= 1;
				end
			end
		endcase
	end
endmodule

