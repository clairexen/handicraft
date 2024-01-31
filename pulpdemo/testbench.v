module testbench;
	// --- Clock and Reset

	reg         clk_i;
	reg         clock_en_i;
	reg         rst_ni;

	initial begin
		#5 clk_i = 0;
		forever begin
			#5 clk_i = 1;
			#5 clk_i = 0;
		end
	end

	initial begin
		rst_ni = 0;
		clock_en_i = 1;
		repeat (10) @(posedge clk_i);
		rst_ni <= 1;
	end
	
	initial begin
		$dumpfile("testbench.vcd");
		$dumpvars(0, testbench);
		repeat (100) @(posedge clk_i);
		$finish;
	end

	// --- Core Config

	reg  [31:0] boot_addr_i  = 32'h 0000_0080;
	reg  [5:0]  cluster_id_i = 0;
	reg  [3:0]  core_id_i    = 0;

	wire        core_busy_o;

	// --- APU

	reg  [4:0]  apu_master_flags_i  = 0;
	reg         apu_master_gnt_i    = 0;
	reg  [31:0] apu_master_result_i = 0;
	reg         apu_master_valid_i  = 0;

	wire [14:0] apu_master_flags_o;
	wire [5:0]  apu_master_op_o;
	wire [95:0] apu_master_operands_o;
	wire        apu_master_ready_o;
	wire        apu_master_req_o;
	wire [0:-1] apu_master_type_o;

	// --- Data R/W Interface

	reg         data_err_i    = 0;
	reg         data_gnt_i    = 0;
	reg  [31:0] data_rdata_i  = 0;
	reg         data_rvalid_i = 0;

	wire [31:0] data_addr_o;
	wire [3:0]  data_be_o;
	wire        data_req_o;
	wire [31:0] data_wdata_o;
	wire        data_we_o;

	// --- Instruction Read Interface

	reg         fetch_enable_i = 1;
	reg         instr_gnt_i    = 0;
	reg  [31:0] instr_rdata_i  = 32'h 0000_0013;
	reg         instr_rvalid_i = 0;

	wire [31:0] instr_addr_o;
	wire        instr_req_o;

	always @* begin
		instr_gnt_i = instr_req_o;
	end

	always @(posedge clk_i) begin
		instr_rvalid_i <= instr_gnt_i && instr_req_o;
	end

	// --- Debug Port

	reg  [14:0] debug_addr_i   = 0;
	reg         debug_halt_i   = 0;
	reg         debug_req_i    = 0;
	reg         debug_resume_i = 0;
	reg  [31:0] debug_wdata_i  = 0;
	reg         debug_we_i     = 0;

	wire        debug_gnt_o;
	wire        debug_halted_o;
	wire [31:0] debug_rdata_o;
	wire        debug_rvalid_o;

	// --- IRQs

	reg         irq_i     = 0;
	reg  [4:0]  irq_id_i  = 0;
	reg         irq_sec_i = 0;

	wire        irq_ack_o;
	wire [4:0]  irq_id_o;

	// --- Stuff (idk what this does)

	reg         test_en_i = 1;
	wire        sec_lvl_o;

	riscv_core uut (
		.apu_master_flags_i   (apu_master_flags_i   ),
		.apu_master_flags_o   (apu_master_flags_o   ),
		.apu_master_gnt_i     (apu_master_gnt_i     ),
		.apu_master_op_o      (apu_master_op_o      ),
		.apu_master_operands_o(apu_master_operands_o),
		.apu_master_ready_o   (apu_master_ready_o   ),
		.apu_master_req_o     (apu_master_req_o     ),
		.apu_master_result_i  (apu_master_result_i  ),
		.apu_master_type_o    (apu_master_type_o    ),
		.apu_master_valid_i   (apu_master_valid_i   ),
		.boot_addr_i          (boot_addr_i          ),
		.clk_i                (clk_i                ),
		.clock_en_i           (clock_en_i           ),
		.cluster_id_i         (cluster_id_i         ),
		.core_busy_o          (core_busy_o          ),
		.core_id_i            (core_id_i            ),
		.data_addr_o          (data_addr_o          ),
		.data_be_o            (data_be_o            ),
		.data_err_i           (data_err_i           ),
		.data_gnt_i           (data_gnt_i           ),
		.data_rdata_i         (data_rdata_i         ),
		.data_req_o           (data_req_o           ),
		.data_rvalid_i        (data_rvalid_i        ),
		.data_wdata_o         (data_wdata_o         ),
		.data_we_o            (data_we_o            ),
		.debug_addr_i         (debug_addr_i         ),
		.debug_gnt_o          (debug_gnt_o          ),
		.debug_halt_i         (debug_halt_i         ),
		.debug_halted_o       (debug_halted_o       ),
		.debug_rdata_o        (debug_rdata_o        ),
		.debug_req_i          (debug_req_i          ),
		.debug_resume_i       (debug_resume_i       ),
		.debug_rvalid_o       (debug_rvalid_o       ),
		.debug_wdata_i        (debug_wdata_i        ),
		.debug_we_i           (debug_we_i           ),
		.ext_perf_counters_i  (                     ),
		.fetch_enable_i       (fetch_enable_i       ),
		.instr_addr_o         (instr_addr_o         ),
		.instr_gnt_i          (instr_gnt_i          ),
		.instr_rdata_i        (instr_rdata_i        ),
		.instr_req_o          (instr_req_o          ),
		.instr_rvalid_i       (instr_rvalid_i       ),
		.irq_ack_o            (irq_ack_o            ),
		.irq_i                (irq_i                ),
		.irq_id_i             (irq_id_i             ),
		.irq_id_o             (irq_id_o             ),
		.irq_sec_i            (irq_sec_i            ),
		.rst_ni               (rst_ni               ),
		.sec_lvl_o            (sec_lvl_o            ),
		.test_en_i            (test_en_i            )
	);
endmodule

module cluster_clock_gating (
	input  clk_i,
	input  en_i,
	input  test_en_i,
	output clk_o
);
	// no clock gates in FPGA flow
	assign clk_o = clk_i;
endmodule
