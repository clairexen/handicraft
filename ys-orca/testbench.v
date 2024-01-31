module testbench;
	reg clk;
	reg reset = 1;
	wire irq = 0;

	initial begin
		#5 clk <= 0;
		repeat (100) #5 clk <= !clk;
		$finish;
	end

	initial begin
		$dumpfile("testbench.vcd");
		$dumpvars(0, testbench);
		repeat (10) @(posedge clk);
		reset <= 0;
	end

	wire [31:0] avm_data_address;
	wire [3:0]  avm_data_byteenable;
	wire        avm_data_read;
	reg  [31:0] avm_data_readdata;
	reg         avm_data_readdatavalid;
	wire        avm_data_waitrequest = 0;
	wire        avm_data_write;
	wire [31:0] avm_data_writedata;

	wire [31:0] avm_instruction_address;
	wire        avm_instruction_read;
	reg  [31:0] avm_instruction_readdata;
	reg         avm_instruction_readdatavalid;
	wire        avm_instruction_waitrequest = 0;

	wire [9:0]  avm_scratch_address = 0;
	wire [3:0]  avm_scratch_byteenable = 0;
	wire        avm_scratch_read = 0;
	wire [31:0] avm_scratch_readdata;
	wire        avm_scratch_readdatavalid;
	wire        avm_scratch_waitrequest;
	wire        avm_scratch_write = 0;
	wire [31:0] avm_scratch_writedata = 0;

	orca uut (
		.clk                          (clk                          ),
		.reset                        (reset                        ),
		.global_interrupts            (irq                          ),

		.avm_data_address             (avm_data_address             ),
		.avm_data_byteenable          (avm_data_byteenable          ),
		.avm_data_read                (avm_data_read                ),
		.avm_data_readdata            (avm_data_readdata            ),
		.avm_data_readdatavalid       (avm_data_readdatavalid       ),
		.avm_data_waitrequest         (avm_data_waitrequest         ),
		.avm_data_write               (avm_data_write               ),
		.avm_data_writedata           (avm_data_writedata           ),

		.avm_instruction_address      (avm_instruction_address      ),
		.avm_instruction_read         (avm_instruction_read         ),
		.avm_instruction_readdata     (avm_instruction_readdata     ),
		.avm_instruction_readdatavalid(avm_instruction_readdatavalid),
		.avm_instruction_waitrequest  (avm_instruction_waitrequest  ),

		.avm_scratch_address          (avm_scratch_address          ),
		.avm_scratch_byteenable       (avm_scratch_byteenable       ),
		.avm_scratch_read             (avm_scratch_read             ),
		.avm_scratch_readdata         (avm_scratch_readdata         ),
		.avm_scratch_readdatavalid    (avm_scratch_readdatavalid    ),
		.avm_scratch_waitrequest      (avm_scratch_waitrequest      ),
		.avm_scratch_write            (avm_scratch_write            ),
		.avm_scratch_writedata        (avm_scratch_writedata        )
	);

	// 1 kB main memory
	reg [31:0] memory [0:255];

	initial begin
		// Simple counter (set to zero and then increment last memory word)
		memory[0] = 32'h 3fc00093; //       li    x1,1020
		memory[1] = 32'h 0000a023; //       sw    x0,0(x1)
		memory[2] = 32'h 0000a103; // loop: lw    x2,0(x1)
		memory[3] = 32'h 00110113; //       addi  x2,x2,1
		memory[4] = 32'h 0020a023; //       sw    x2,0(x1)
		memory[5] = 32'h ff5ff06f; //       j     <loop>
	end
	
	always @(posedge clk) begin
		avm_instruction_readdatavalid <= 0;
		avm_data_readdatavalid <= 0;
		if (!reset) begin
			if (avm_instruction_read) begin
				$display("IRD: %08x %08x", avm_instruction_address, memory[avm_instruction_address >> 2]);
				avm_instruction_readdata <= memory[avm_instruction_address >> 2];
				avm_instruction_readdatavalid <= 1;
			end
			if (avm_data_read) begin
				$display("DRD: %08x %08x", avm_data_address, memory[avm_data_address >> 2]);
				avm_data_readdata <= memory[avm_data_address >> 2];
				avm_data_readdatavalid <= 1;
			end
			if (avm_data_write) begin
				$display("DWR: %08x %08x %b", avm_data_address, avm_data_writedata, avm_data_byteenable);
				if (avm_data_byteenable[0]) memory[avm_data_address >> 2][ 7: 0] <= avm_data_writedata[ 7: 0];
				if (avm_data_byteenable[1]) memory[avm_data_address >> 2][15: 0] <= avm_data_writedata[15: 0];
				if (avm_data_byteenable[2]) memory[avm_data_address >> 2][23:16] <= avm_data_writedata[23:16];
				if (avm_data_byteenable[3]) memory[avm_data_address >> 2][31:24] <= avm_data_writedata[31:24];
			end
		end
	end
endmodule
