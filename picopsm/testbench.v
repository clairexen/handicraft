
module testbench;

	reg [7:0] mem [0:65535];

	reg clk, resetn;

	wire        mem_axi_awvalid;
	wire        mem_axi_awready;
	wire [15:0] mem_axi_awaddr;
	wire [ 2:0] mem_axi_awprot;

	wire        mem_axi_wvalid;
	wire        mem_axi_wready;
	wire [ 7:0] mem_axi_wdata;

	reg         mem_axi_bvalid;
	wire        mem_axi_bready;

	wire        mem_axi_arvalid;
	wire        mem_axi_arready;
	wire [15:0] mem_axi_araddr;
	wire [ 2:0] mem_axi_arprot;

	reg         mem_axi_rvalid;
	wire        mem_axi_rready;
	reg  [ 7:0] mem_axi_rdata;

	picopsm uut (
		.clk(clk),
		.resetn(resetn),

		.mem_axi_awvalid(mem_axi_awvalid),
		.mem_axi_awready(mem_axi_awready),
		.mem_axi_awaddr(mem_axi_awaddr),
		.mem_axi_awprot(mem_axi_awprot),

		.mem_axi_wvalid(mem_axi_wvalid),
		.mem_axi_wready(mem_axi_wready),
		.mem_axi_wdata(mem_axi_wdata),

		.mem_axi_bvalid(mem_axi_bvalid),
		.mem_axi_bready(mem_axi_bready),

		.mem_axi_arvalid(mem_axi_arvalid),
		.mem_axi_arready(mem_axi_arready),
		.mem_axi_araddr(mem_axi_araddr),
		.mem_axi_arprot(mem_axi_arprot),

		.mem_axi_rvalid(mem_axi_rvalid),
		.mem_axi_rready(mem_axi_rready),
		.mem_axi_rdata(mem_axi_rdata)
	);

        initial begin
		$dumpfile("testbench.vcd");
		$dumpvars(0, testbench);
		`include "example_001.v"
	end

	initial begin
		clk = 1;
		forever #5 clk = ~clk;
	end

	integer i;

	initial begin
		resetn <= 0;
		mem_axi_bvalid <= 0;
		mem_axi_rvalid <= 0;
		mem_axi_rdata <= 'bx;
		repeat (100) @(posedge clk);
		resetn <= 1;
		repeat (500) @(posedge clk);
		for (i = 0; i < 256; i = i + 16)
			$display("%x: %x %x %x %x  %x %x %x %x  %x %x %x %x  %x %x %x %x", i,
				mem[i+0], mem[i+1], mem[i+2], mem[i+3], mem[i+4], mem[i+5], mem[i+6], mem[i+7],
				mem[i+8], mem[i+9], mem[i+10], mem[i+11], mem[i+12], mem[i+13], mem[i+14], mem[i+15]);
		$finish;
	end

	assign mem_axi_arready = mem_axi_arvalid && !mem_axi_rvalid;
	assign mem_axi_awready = mem_axi_awvalid && mem_axi_wvalid && !mem_axi_bvalid;
	assign mem_axi_wready = mem_axi_awready;

	always @(posedge clk) begin
		if (resetn) begin
			if (mem_axi_awready) begin
				$display("WR %x: %x", mem_axi_awaddr, mem_axi_wdata);
				mem[mem_axi_awaddr] <= mem_axi_wdata;
				mem_axi_bvalid <= 1;
			end
			if (mem_axi_bvalid && mem_axi_bready)
				mem_axi_bvalid <= 0;
			if (mem_axi_arready) begin
				$display("RD %x: %x", mem_axi_araddr, mem[mem_axi_araddr]);
				mem_axi_rdata <= mem[mem_axi_araddr];
				mem_axi_rvalid <= 1;
			end
			if (mem_axi_rvalid && mem_axi_rready)
				mem_axi_rvalid <= 0;
		end
	end
endmodule
