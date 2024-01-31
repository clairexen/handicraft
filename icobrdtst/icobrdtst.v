// PMOD support cores
`define PMOD1_CORE gpio
`define PMOD2_CORE gpio
`define PMOD3_CORE gpio
`define PMOD4_CORE gpio

// Divide the 12 MHz by this power of two: 0=12MHz, 1=6MHz, 2=3MHz, ...
`define POW2CLOCKDIV 1

module icobrdtst (
	input CLK12MHZ,
	output reg LED1, LED2, LED3,

	// RasPi Interface: 9 Data Lines (cmds have MSB set)
	inout RASPI_11, RASPI_12, RASPI_15, RASPI_16, RASPI_19, RASPI_21, RASPI_24, RASPI_35, RASPI_36,

	// RasPi Interface: Control Lines
	input RASPI_38, RASPI_40,

	// PMODS
	inout PMOD1_1, PMOD1_2, PMOD1_3, PMOD1_4, PMOD1_7, PMOD1_8, PMOD1_9, PMOD1_10,
	inout PMOD2_1, PMOD2_2, PMOD2_3, PMOD2_4, PMOD2_7, PMOD2_8, PMOD2_9, PMOD2_10,
	inout PMOD3_1, PMOD3_2, PMOD3_3, PMOD3_4, PMOD3_7, PMOD3_8, PMOD3_9, PMOD3_10,
	inout PMOD4_1, PMOD4_2, PMOD4_3, PMOD4_4, PMOD4_7, PMOD4_8, PMOD4_9, PMOD4_10
);
	// 2048 32bit words = 8k bytes memory
	// 1024 32bit words = 4k bytes memory
	// 128 32bit words = 512 bytes memory
	localparam BOOT_MEM_SIZE = 1024;

	// -------------------------------
	// PLL

	wire clk, clk90;
	wire resetn;

	ico_clkgen clkgen (
		.CLK12MHZ(CLK12MHZ),
		.clk(clk),
		.clk90(clk90),
		.resetn(resetn)
	);


	// -------------------------------
	// PMODs

	reg pmod1_ctrl_wr;
	reg pmod1_ctrl_rd;
	reg [ 7:0] pmod1_ctrl_addr;
	reg [31:0] pmod1_ctrl_wdat;
	wire [31:0] pmod1_ctrl_rdat;
	wire pmod1_ctrl_done;

	pmod_`PMOD1_CORE pmod1 (
		.clk       (clk             ),
		.resetn    (resetn          ),
		.ctrl_wr   (pmod1_ctrl_wr   ),
		.ctrl_rd   (pmod1_ctrl_rd   ),
		.ctrl_addr (pmod1_ctrl_addr ),
		.ctrl_wdat (pmod1_ctrl_wdat ),
		.ctrl_rdat (pmod1_ctrl_rdat ),
		.ctrl_done (pmod1_ctrl_done ),

		.PMOD_1    (PMOD1_1         ),
		.PMOD_2    (PMOD1_2         ),
		.PMOD_3    (PMOD1_3         ),
		.PMOD_4    (PMOD1_4         ),
		.PMOD_7    (PMOD1_7         ),
		.PMOD_8    (PMOD1_8         ),
		.PMOD_9    (PMOD1_9         ),
		.PMOD_10   (PMOD1_10        ),

		.PMOD2_1   (PMOD2_1         ),
		.PMOD2_2   (PMOD2_2         ),
		.PMOD2_3   (PMOD2_3         ),
		.PMOD2_4   (PMOD2_4         ),
		.PMOD2_7   (PMOD2_7         ),
		.PMOD2_8   (PMOD2_8         ),
		.PMOD2_9   (PMOD2_9         ),
		.PMOD2_10  (PMOD2_10        )
	);

	reg pmod2_ctrl_wr;
	reg pmod2_ctrl_rd;
	reg [ 7:0] pmod2_ctrl_addr;
	reg [31:0] pmod2_ctrl_wdat;
	wire [31:0] pmod2_ctrl_rdat;
	wire pmod2_ctrl_done;

	pmod_`PMOD2_CORE pmod2 (
		.clk       (clk             ),
		.resetn    (resetn          ),
		.ctrl_wr   (pmod2_ctrl_wr   ),
		.ctrl_rd   (pmod2_ctrl_rd   ),
		.ctrl_addr (pmod2_ctrl_addr ),
		.ctrl_wdat (pmod2_ctrl_wdat ),
		.ctrl_rdat (pmod2_ctrl_rdat ),
		.ctrl_done (pmod2_ctrl_done ),

		.PMOD_1    (PMOD2_1         ),
		.PMOD_2    (PMOD2_2         ),
		.PMOD_3    (PMOD2_3         ),
		.PMOD_4    (PMOD2_4         ),
		.PMOD_7    (PMOD2_7         ),
		.PMOD_8    (PMOD2_8         ),
		.PMOD_9    (PMOD2_9         ),
		.PMOD_10   (PMOD2_10        ),

		.PMOD2_1   (PMOD1_1         ),
		.PMOD2_2   (PMOD1_2         ),
		.PMOD2_3   (PMOD1_3         ),
		.PMOD2_4   (PMOD1_4         ),
		.PMOD2_7   (PMOD1_7         ),
		.PMOD2_8   (PMOD1_8         ),
		.PMOD2_9   (PMOD1_9         ),
		.PMOD2_10  (PMOD1_10        )
	);

	reg pmod3_ctrl_wr;
	reg pmod3_ctrl_rd;
	reg [ 7:0] pmod3_ctrl_addr;
	reg [31:0] pmod3_ctrl_wdat;
	wire [31:0] pmod3_ctrl_rdat;
	wire pmod3_ctrl_done;

	pmod_`PMOD3_CORE pmod3 (
		.clk       (clk             ),
		.resetn    (resetn          ),
		.ctrl_wr   (pmod3_ctrl_wr   ),
		.ctrl_rd   (pmod3_ctrl_rd   ),
		.ctrl_addr (pmod3_ctrl_addr ),
		.ctrl_wdat (pmod3_ctrl_wdat ),
		.ctrl_rdat (pmod3_ctrl_rdat ),
		.ctrl_done (pmod3_ctrl_done ),

		.PMOD_1    (PMOD3_1         ),
		.PMOD_2    (PMOD3_2         ),
		.PMOD_3    (PMOD3_3         ),
		.PMOD_4    (PMOD3_4         ),
		.PMOD_7    (PMOD3_7         ),
		.PMOD_8    (PMOD3_8         ),
		.PMOD_9    (PMOD3_9         ),
		.PMOD_10   (PMOD3_10        ),

		.PMOD2_1   (PMOD4_1         ),
		.PMOD2_2   (PMOD4_2         ),
		.PMOD2_3   (PMOD4_3         ),
		.PMOD2_4   (PMOD4_4         ),
		.PMOD2_7   (PMOD4_7         ),
		.PMOD2_8   (PMOD4_8         ),
		.PMOD2_9   (PMOD4_9         ),
		.PMOD2_10  (PMOD4_10        )
	);

	reg pmod4_ctrl_wr;
	reg pmod4_ctrl_rd;
	reg [ 7:0] pmod4_ctrl_addr;
	reg [31:0] pmod4_ctrl_wdat;
	wire [31:0] pmod4_ctrl_rdat;
	wire pmod4_ctrl_done;

	pmod_`PMOD4_CORE pmod4 (
		.clk       (clk             ),
		.resetn    (resetn          ),
		.ctrl_wr   (pmod4_ctrl_wr   ),
		.ctrl_rd   (pmod4_ctrl_rd   ),
		.ctrl_addr (pmod4_ctrl_addr ),
		.ctrl_wdat (pmod4_ctrl_wdat ),
		.ctrl_rdat (pmod4_ctrl_rdat ),
		.ctrl_done (pmod4_ctrl_done ),

		.PMOD_1    (PMOD4_1         ),
		.PMOD_2    (PMOD4_2         ),
		.PMOD_3    (PMOD4_3         ),
		.PMOD_4    (PMOD4_4         ),
		.PMOD_7    (PMOD4_7         ),
		.PMOD_8    (PMOD4_8         ),
		.PMOD_9    (PMOD4_9         ),
		.PMOD_10   (PMOD4_10        ),

		.PMOD2_1   (PMOD3_1         ),
		.PMOD2_2   (PMOD3_2         ),
		.PMOD2_3   (PMOD3_3         ),
		.PMOD2_4   (PMOD3_4         ),
		.PMOD2_7   (PMOD3_7         ),
		.PMOD2_8   (PMOD3_8         ),
		.PMOD2_9   (PMOD3_9         ),
		.PMOD2_10  (PMOD3_10        )
	);


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

	// send ep1: unused
	wire send_ep1_valid = 0;
	wire send_ep1_ready;
	wire [7:0] send_ep1_data = 'bx;

	// send ep2: console output
	reg  send_ep2_valid;
	wire send_ep2_ready;
	reg  [7:0] send_ep2_data;

	// send ep3: unused
	wire send_ep3_valid = 0;
	wire send_ep3_ready;
	wire [7:0] send_ep3_data = 'bx;

	// trigger lines
	wire trigger_0;  // unused
	wire trigger_1;  // unused
	wire trigger_2;  // unused
	wire trigger_3;  // unused

	ico_raspi_interface #(
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
	// Memory/IO Interface

	reg [31:0] memory [0:BOOT_MEM_SIZE-1];
	initial $readmemh("firmware.hex", memory);

	always @(posedge clk) begin
		mem_ready <= 0;

		pmod1_ctrl_wr <= 0;
		pmod2_ctrl_wr <= 0;
		pmod3_ctrl_wr <= 0;
		pmod4_ctrl_wr <= 0;

		pmod1_ctrl_rd <= 0;
		pmod2_ctrl_rd <= 0;
		pmod3_ctrl_rd <= 0;
		pmod4_ctrl_rd <= 0;

		pmod1_ctrl_addr <= mem_addr[7:0];
		pmod2_ctrl_addr <= mem_addr[7:0];
		pmod3_ctrl_addr <= mem_addr[7:0];
		pmod4_ctrl_addr <= mem_addr[7:0];

		pmod1_ctrl_wdat <= mem_wdata;
		pmod2_ctrl_wdat <= mem_wdata;
		pmod3_ctrl_wdat <= mem_wdata;
		pmod4_ctrl_wdat <= mem_wdata;

		if (send_ep2_ready)
			send_ep2_valid <= 0;

		recv_ep2_ready <= 0;

		if (!resetn_picorv32) begin
			LED1 <= 0;
			LED2 <= 0;
			LED3 <= 0;

			send_ep2_valid <= 0;

			if (prog_mem_active) begin
				memory[prog_mem_addr] <= prog_mem_data;
			end
		end else
		if (mem_valid && !mem_ready) begin
			(* parallel_case *)
			case (1)
				(mem_addr >> 2) < BOOT_MEM_SIZE: begin
					if (mem_wstrb) begin
						if (mem_wstrb[0]) memory[mem_addr >> 2][ 7: 0] <= mem_wdata[ 7: 0];
						if (mem_wstrb[1]) memory[mem_addr >> 2][15: 8] <= mem_wdata[15: 8];
						if (mem_wstrb[2]) memory[mem_addr >> 2][23:16] <= mem_wdata[23:16];
						if (mem_wstrb[3]) memory[mem_addr >> 2][31:24] <= mem_wdata[31:24];
					end else begin
						mem_rdata <= memory[mem_addr >> 2];
					end
					mem_ready <= 1;
				end
				(mem_addr & 32'hF000_0000) == 32'h2000_0000: begin
					mem_ready <= 1;
					mem_rdata <= 0;
					if (mem_wstrb) begin
						if (mem_addr[15:8] == 0) begin
							if (mem_addr[7:0] == 8'h 00) {LED3, LED2, LED1} <= mem_wdata;
						end
						if (mem_addr[15:8] == 1) begin
							mem_ready <= pmod1_ctrl_done;
							pmod1_ctrl_wr <= 1;
						end
						if (mem_addr[15:8] == 2) begin
							mem_ready <= pmod2_ctrl_done;
							pmod2_ctrl_wr <= 1;
						end
						if (mem_addr[15:8] == 3) begin
							mem_ready <= pmod3_ctrl_done;
							pmod3_ctrl_wr <= 1;
						end
						if (mem_addr[15:8] == 4) begin
							mem_ready <= pmod4_ctrl_done;
							pmod4_ctrl_wr <= 1;
						end
					end else begin
						if (mem_addr[15:8] == 0) begin
							if (mem_addr[7:0] == 8'h 00) mem_rdata <= {LED3, LED2, LED1};
						end
						if (mem_addr[15:8] == 1) begin
							mem_ready <= pmod1_ctrl_done;
							pmod1_ctrl_rd <= !pmod1_ctrl_done;
							mem_rdata <= pmod1_ctrl_rdat;
						end
						if (mem_addr[15:8] == 2) begin
							mem_ready <= pmod2_ctrl_done;
							pmod2_ctrl_rd <= !pmod2_ctrl_done;
							mem_rdata <= pmod2_ctrl_rdat;
						end
						if (mem_addr[15:8] == 3) begin
							mem_ready <= pmod3_ctrl_done;
							pmod3_ctrl_rd <= !pmod3_ctrl_done;
							mem_rdata <= pmod3_ctrl_rdat;
						end
						if (mem_addr[15:8] == 4) begin
							mem_ready <= pmod4_ctrl_done;
							pmod4_ctrl_rd <= !pmod4_ctrl_done;
							mem_rdata <= pmod4_ctrl_rdat;
						end
					end
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

module ico_clkgen (
	input CLK12MHZ,
	output clk, clk90,
	output resetn
);
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

module ico_crossclkfifo #(
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

module ico_raspi_interface #(
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

	ico_crossclkfifo #(
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
		.PULLUP(1'b 1) // <- enable pullup
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

