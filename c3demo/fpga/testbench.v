module testbench;
	reg clk = 0;
	integer i, k;

	wire [8:0] raspi_dat;
	reg [8:0] raspi_dout = 9'bz;
	reg raspi_dir = 0, raspi_clk = 0;
	assign raspi_dat = raspi_dout;

	wire [15:0] sram_a, sram_d;
	wire sram_ce, sram_we, sram_oe, sram_lb, sram_ub;
	reg [15:0] sram_memory [0:65535];
	reg [15:0] sram_dout;

	assign sram_d = sram_dout;

	always @(sram_we, sram_ce, sram_oe, sram_lb, sram_ub, sram_a, sram_d) begin
		// Truth table on page 2 of the SRAM data sheet:
		// http://www.issi.com/WW/pdf/61-64WV6416DAxx-DBxx.pdf
		casez ({sram_we, sram_ce, sram_oe, sram_lb, sram_ub})
			// Not Selected / Output Disabled
			5'b z1zzz, 5'b 101zz, 5'b z0z11: sram_dout = 16'bz;
			// Read
			5'b 10001: sram_dout = {8'bz, sram_memory[sram_a][7:0]};
			5'b 10010: sram_dout = {sram_memory[sram_a][15:8], 8'bz};
			5'b 10000: sram_dout = sram_memory[sram_a];
			// Write
			5'b 00z01: sram_memory[sram_a][7:0] = sram_d[7:0];
			5'b 00z10: sram_memory[sram_a][15:8] = sram_d[15:8];
			5'b 00z00: sram_memory[sram_a] = sram_d;
		endcase
	end

	initial begin
		#5; forever #5 clk = !clk;
	end

	c3demo uut (
		.CLK12MHZ(clk),
		.RASPI_11(raspi_dat[8]),
		.RASPI_12(raspi_dat[7]),
		.RASPI_15(raspi_dat[6]),
		.RASPI_16(raspi_dat[5]),
		.RASPI_19(raspi_dat[4]),
		.RASPI_21(raspi_dat[3]),
		.RASPI_24(raspi_dat[2]),
		.RASPI_35(raspi_dat[1]),
		.RASPI_36(raspi_dat[0]),
		.RASPI_38(raspi_dir),
		.RASPI_40(raspi_clk),

		.SRAM_A0(sram_a[0]),
		.SRAM_A1(sram_a[1]),
		.SRAM_A2(sram_a[2]),
		.SRAM_A3(sram_a[3]),
		.SRAM_A4(sram_a[4]),
		.SRAM_A5(sram_a[5]),
		.SRAM_A6(sram_a[6]),
		.SRAM_A7(sram_a[7]),
		.SRAM_A8(sram_a[8]),
		.SRAM_A9(sram_a[9]),
		.SRAM_A10(sram_a[10]),
		.SRAM_A11(sram_a[11]),
		.SRAM_A12(sram_a[12]),
		.SRAM_A13(sram_a[13]),
		.SRAM_A14(sram_a[14]),
		.SRAM_A15(sram_a[15]),

		.SRAM_D0(sram_d[0]),
		.SRAM_D1(sram_d[1]),
		.SRAM_D2(sram_d[2]),
		.SRAM_D3(sram_d[3]),
		.SRAM_D4(sram_d[4]),
		.SRAM_D5(sram_d[5]),
		.SRAM_D6(sram_d[6]),
		.SRAM_D7(sram_d[7]),
		.SRAM_D8(sram_d[8]),
		.SRAM_D9(sram_d[9]),
		.SRAM_D10(sram_d[10]),
		.SRAM_D11(sram_d[11]),
		.SRAM_D12(sram_d[12]),
		.SRAM_D13(sram_d[13]),
		.SRAM_D14(sram_d[14]),
		.SRAM_D15(sram_d[15]),

		.SRAM_CE(sram_ce),
		.SRAM_WE(sram_we),
		.SRAM_OE(sram_oe),
		.SRAM_LB(sram_lb),
		.SRAM_UB(sram_ub)
	);

	integer f;
	reg [8:0] a, b, c;
	reg [31:0] fw;

	task raspi_send;
		input [8:0] word;
		begin
			raspi_dir <= 1;
			raspi_dout <= word;
			#20;
			raspi_clk <= 1;
			#20;
			raspi_clk <= 0;
		end
	endtask

	task raspi_recv;
		output [8:0] word;
		begin
			raspi_dir <= 0;
			raspi_dout <= 'bz;
			#20;
			word <= raspi_dat;
			raspi_clk <= 1;
			#20;
			raspi_clk <= 0;
		end
	endtask

	initial begin
		$dumpfile("testbench.vcd");
		$dumpvars(0, testbench);

		repeat (100)
			@(posedge clk);

		raspi_send(9'h 1ff);
		raspi_send(9'h 0ff);

		b = 0;
		while (b != 9'h 1ff)
			raspi_recv(b);

		raspi_send(9'h 100);

		for (a = 64; a < 128; a = a+1)
			raspi_send(a);

		c = 9'h 100;
		raspi_recv(b);
		$display("Link test: BEGIN  %x (expected: %x, %0s)", b, c, b === c ? "ok" : "ERROR");
		if (b !== c) $finish;

		for (a = 64; a < 128; a = a+1) begin
			raspi_recv(b);
			c =  (((a << 5) + a) ^ 7) & 255;
			$display("Link test: %x -> %x (expected: %x, %0s)", a, b, c, b === c ? "ok" : "ERROR");
			if (b !== c) $finish;
		end

		c = 9'h 1ff;
		raspi_recv(b);
		$display("Link test: END    %x (expected: %x, %0s)", b, c, b === c ? "ok" : "ERROR");
		if (b !== c) $finish;

		repeat (1000)
			@(posedge clk);

		$display("Uploading firmware..");

		raspi_send(9'h 1ff);
		raspi_send(9'h 0ff);

		f = $fopen("firmware.hex", "r");
		raspi_send(9'h 101);

		while ($fscanf(f, "%x", fw) == 1) begin
			raspi_send(fw[ 7: 0]);
			raspi_send(fw[15: 8]);
			raspi_send(fw[23:16]);
			raspi_send(fw[31:24]);
		end

		$fclose(f);

		raspi_send(9'h 1ff);
		raspi_send(9'h 0ff);

		b = 0;
		while (b != 9'h 1ff)
			raspi_recv(b);

		$display("Reading debugger..");

		raspi_send(9'h 1ff);
		raspi_send(9'h 000);

		repeat (100)
			raspi_recv(b);
		while (b != 9'h 1ff)
			raspi_recv(b);

		$display("Running the system..");

		for (k = 0; k < 10; k = k+1) begin
			$write("%3d:", k);
			for (i = 0; i < 30; i = i+1) begin
				repeat (10000) @(posedge clk);
				$write("%3d", i);
				$fflush;
			end
			$display("");
		end
		$finish;
	end
endmodule
