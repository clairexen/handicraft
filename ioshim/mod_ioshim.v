// IOSHIM Register File:
//
//   0x0000 .. 0x0fff ioshim memory
//      read and write (of any width) is supported
//
//   0x1000 .. 0x1003 ioshim reset register
//      write -1 = hold ioshim in reset
//      write other value = start exec at addr
//      only 32 bit write are supported
//      reading of this register is not supported
//
module icosoc_mod_ioshim #(
	parameter integer CLOCK_FREQ_HZ = 0,
	parameter integer MEMSIZE = 128
) (
	input clk,
	input resetn,

	input [3:0] ctrl_wr,
	input ctrl_rd,
	input [15:0] ctrl_addr,
	input [31:0] ctrl_wdat,
	output reg [31:0] ctrl_rdat,
	output reg ctrl_done,

`ifdef ICOSOC
	inout [7:0] gpio
`else
	output [7:0] gpio_dir,
	output [7:0] gpio_dout,
	input [7:0] gpio_din
`endif
);
`ifdef ICOSOC
	wire [7:0] gpio_dir;
	wire [7:0] gpio_dout;
	wire [7:0] gpio_din;

	SB_IO #(
		.PIN_TYPE(6'b 1010_01),
		.PULLUP(1'b 0)
	) ios [7:0] (
		.PACKAGE_PIN(gpio),
		.OUTPUT_ENABLE(gpio_dir),
		.D_OUT_0(gpio_dout),
		.D_IN_0(gpio_din)
	);
`endif

	reg memio_rd;
	reg [1:0] memio_wr;
	reg [10:0] memio_addr;
	reg [15:0] memio_wdata;
	wire [15:0] memio_rdata;
	wire memio_done;

	wire cpu_io_en;
	wire [4:0] cpu_io_epnum;
	wire [7:0] cpu_io_dout1;
	wire [7:0] cpu_io_dout2;
	wire [15:0] cpu_io_ab_dout;

	wire cpu_io_wreg, cpu_io_wa, cpu_io_wb;
	wire [7:0] cpu_io_din;
	wire [15:0] cpu_io_ab_din;

	wire gpio_ep_en = cpu_io_en && (cpu_io_epnum == 0);
	wire [7:0] gpio_ep_dout1 = cpu_io_dout1;
	wire [7:0] gpio_ep_dout2 = cpu_io_dout2;
	wire [15:0] gpio_ep_ab_dout = cpu_io_ab_dout;

	wire gpio_ep_wreg, gpio_ep_wa, gpio_ep_wb;
	wire [7:0] gpio_ep_din;
	wire [15:0] gpio_ep_ab_din;

	assign cpu_io_wreg = |{gpio_ep_wreg};
	assign cpu_io_wa = |{gpio_ep_wa};
	assign cpu_io_wb = |{gpio_ep_wb};
	assign cpu_io_din = gpio_ep_wreg ? gpio_ep_din : 'bx;
	assign cpu_io_ab_din = (gpio_ep_wa || gpio_ep_wb) ? gpio_ep_ab_din : 'bx;

	reg cpu_running;
	wire cpu_resetn = resetn && cpu_running;

	reg ctrl_state;
	always @(posedge clk) begin
		if (!resetn) begin
			ctrl_done <= 0;
			ctrl_state <= 0;
			cpu_running <= 0;
			memio_rd <= 0;
			memio_wr <= 0;
		end else begin
			if (ctrl_wr && !ctrl_done) begin
				if (ctrl_addr == 'h 1000) begin
					// perform ioshim reset
					if (!ctrl_state) begin
						cpu_running <= 0;
						ctrl_state <= 1;
					end else begin
						cpu_running <= !ctrl_wdat[31];
						ctrl_state <= 0;
						ctrl_done <= 1;
					end
				end else begin
					// write to ioshim memory
					if (!ctrl_state) begin
						memio_wr <= ctrl_wr[1:0];
						memio_addr <= ctrl_addr >> 1;
						memio_wdata <= ctrl_wdat[15:0];
						if (memio_done || !ctrl_wr[1:0]) begin
							memio_wr <= ctrl_wr[3:2];
							memio_addr <= (ctrl_addr >> 1) ^ 1;
							memio_wdata <= ctrl_wdat[31:16];
							ctrl_state <= 1;
						end
					end else begin
						if (memio_done || !memio_wr) begin
							memio_wr <= 0;
							ctrl_state <= 0;
							ctrl_done <= 1;
						end
					end
				end
			end else
			if (ctrl_rd && !ctrl_done) begin
				if (!ctrl_state) begin
					memio_rd <= 1;
					memio_addr <= ctrl_addr >> 1;
					if (memio_done) begin
						memio_addr <= (ctrl_addr >> 1) ^ 1;
						ctrl_rdat[15:0] <= memio_rdata;
						ctrl_state <= 1;
					end
				end else begin
					if (memio_done) begin
						memio_rd <= 0;
						ctrl_rdat[31:16] <= memio_rdata;
						ctrl_state <= 0;
						ctrl_done <= 1;
					end
				end
			end else begin
				ctrl_done <= 0;
				ctrl_state <= 0;
				memio_rd <= 0;
				memio_wr <= 0;
			end
		end
	end

	ioshim_cpu #(
		.MEMSIZE(MEMSIZE)
	) ioshim_cpu (
		.clk          (clk            ),
		.resetn       (cpu_resetn     ),
		.rstaddr      (ctrl_wdat[10:0]),

		.memio_rd     (memio_rd       ),
		.memio_wr     (memio_wr       ),
		.memio_addr   (memio_addr     ),
		.memio_wdata  (memio_wdata    ),
		.memio_rdata  (memio_rdata    ),
		.memio_done   (memio_done     ),

		.io_en        (cpu_io_en      ),
		.io_epnum     (cpu_io_epnum   ),
		.io_dout1     (cpu_io_dout1   ),
		.io_dout2     (cpu_io_dout2   ),
		.io_ab_dout   (cpu_io_ab_dout ),
		.io_wreg      (cpu_io_wreg    ),
		.io_wa        (cpu_io_wa      ),
		.io_wb        (cpu_io_wb      ),
		.io_din       (cpu_io_din     ),
		.io_ab_din    (cpu_io_ab_din  )
	);

	ioshim_gpio ioshim_gpio (
		.clk       (clk            ),
		.resetn    (resetn         ),

		.io_en     (gpio_ep_en     ),
		.io_dout1  (gpio_ep_dout1  ),
		.io_dout2  (gpio_ep_dout2  ),
		.io_ab_dout(gpio_ep_ab_dout),
		.io_wreg   (gpio_ep_wreg   ),
		.io_wa     (gpio_ep_wa     ),
		.io_wb     (gpio_ep_wb     ),
		.io_din    (gpio_ep_din    ),
		.io_ab_din (gpio_ep_ab_din ),

		.gpio_dir  (gpio_dir       ),
		.gpio_dout (gpio_dout      ),
		.gpio_din  (gpio_din       )
	);
endmodule

