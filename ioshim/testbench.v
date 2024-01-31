module testbench;
	reg clk = 1;
	reg resetn = 0;

	always #5 clk = ~clk;

	initial begin
		repeat (100) @(posedge clk);
		resetn <= 1;
	end

	reg [1023:0] meminit_value;

	initial begin
		if ($value$plusargs("meminit=%s", meminit_value)) begin
			$readmemh(meminit_value, uut.mem.memory);
		end
		if ($test$plusargs("vcd")) begin
			$dumpfile("testbench.vcd");
			$dumpvars(0, testbench);
		end
		repeat (150) @(posedge clk);
		$finish;
	end

	reg memio_rd = 0;
	reg [1:0] memio_wr = 0;
	reg [10:0] memio_addr = 'bx;
	reg [15:0] memio_wdata = 'bx;
	wire [15:0] memio_rdata;
	wire memio_done;

	wire io_en;
	wire [4:0] io_epnum;
	wire [7:0] io_dout1;
	wire [7:0] io_dout2;
	wire [15:0] io_ab_dout;
	reg io_wreg, io_wa, io_wb;
	reg [7:0] io_din;
	reg [15:0] io_ab_din;

	ioshim_cpu #(
		.MEMSIZE(512)
	) uut (
		.clk          (clk          ),
		.resetn       (resetn       ),
		.rstaddr      (11'b0        ),

		.memio_rd     (memio_rd     ),
		.memio_wr     (memio_wr     ),
		.memio_addr   (memio_addr   ),
		.memio_wdata  (memio_wdata  ),
		.memio_rdata  (memio_rdata  ),
		.memio_done   (memio_done   ),

		.io_en        (io_en        ),
		.io_epnum     (io_epnum     ),
		.io_dout1     (io_dout1     ),
		.io_dout2     (io_dout2     ),
		.io_ab_dout   (io_ab_dout   ),
		.io_wreg      (io_wreg      ),
		.io_wa        (io_wa        ),
		.io_wb        (io_wb        ),
		.io_din       (io_din       ),
		.io_ab_din    (io_ab_din    )
	);

	always @(posedge clk) begin
		if (io_en) begin
			io_din <= 'bx;
			io_wreg <= 0;
			case (io_epnum)
				1: begin
					io_din <= io_dout2 + 1;
					io_wreg <= 1;
				end
			endcase
			$display("io%1d %1d", io_epnum, io_dout2);
		end else begin
			io_din <= 'bx;
			io_wreg <= 'bx;
		end
	end
endmodule
