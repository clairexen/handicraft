`timescale 1 ns / 1 ps

module chip_tb;
	localparam integer SIM_CYCLES = 1000000;
	localparam real CLK_PERIOD = 1000.0 / 12.3;
	// localparam real SCLK_PERIOD = CLK_PERIOD * 0.4854;
	// localparam real WORD_SEP = 0;
	localparam real SCLK_PERIOD = 1000.0;
	localparam real WORD_SEP = 10.0;

	reg clk = 1;
	always #(CLK_PERIOD/2.0) clk = ~clk;

	reg spi_sclk;
	reg spi_mosi;
	wire spi_miso;
	reg [7:0] spi_csel;
	reg [7:0] spi_data;

	wire [7:0] pmod_my_pwm_0_A;
	wire [7:0] pmod_my_pwm_1_B;
	wire [7:0] pmod_my_pwm_2_C;
	wire [7:0] pmod_my_pwm_3_D;

	chip uut (
		.clk(clk),
		.spi_sclk(spi_sclk),
		.spi_mosi(spi_mosi),
		.spi_miso(spi_miso),
		.spi_csel(spi_csel),
		.pmod_my_pwm_0_A(pmod_my_pwm_0_A),
		.pmod_my_pwm_1_B(pmod_my_pwm_1_B),
		.pmod_my_pwm_2_C(pmod_my_pwm_2_C),
		.pmod_my_pwm_3_D(pmod_my_pwm_3_D)
	);

	task do_spi_csel;
		input [7:0] new_csel;
		begin
			#(CLK_PERIOD*10);
			spi_sclk <= 1;
			spi_csel <= new_csel;
			$display("SPI: Select EP %1d", new_csel);
			#(CLK_PERIOD*10 + WORD_SEP);
		end
	endtask

	task do_spi_xfer;
		input [7:0] din;
		integer i;
		begin
			spi_data = din;
			for (i = 0; i < 8; i = i+1) begin
				spi_sclk <= 0;
				spi_mosi <= spi_data[7];
				#(SCLK_PERIOD/2.0);
				spi_sclk <= 1;
				spi_data = {spi_data[6:0], spi_miso};
				#(SCLK_PERIOD/2.0);
			end
			$display("SPI: I=0x%02x O=0x%02x", din, spi_data);
			#(WORD_SEP);
		end
	endtask

	task do_spi_check;
		input [7:0] dout;
		begin
			if (spi_data !== dout) begin
				$display("SPI: Unexpected Output (expected 0x%02x)", dout);
				$stop;
			end
		end
	endtask

	integer i, j;
	initial begin
		// SPI MODE: CPOL=1, CPHA=1

		do_spi_csel(0);
		repeat (50) @(posedge clk);
		do_spi_xfer(00);
		repeat (50) @(posedge clk);

		do_spi_csel(1);
		do_spi_xfer('h 77);
		do_spi_check(143);
		do_spi_xfer('h 88);
		do_spi_check(42);
		do_spi_xfer('h 99);
		do_spi_check('h 77);
		do_spi_xfer('h 00);
		do_spi_check('h 88);
		do_spi_xfer('h 07);
		do_spi_check('h 99);
		do_spi_csel(0);

		do_spi_csel(1);
		do_spi_xfer('h 77);
		do_spi_check(143);
		do_spi_xfer('h 88);
		do_spi_check(42);
		do_spi_xfer('h 99);
		do_spi_check('h 77);
		do_spi_xfer('h 00);
		do_spi_check('h 88);
		do_spi_xfer('h 07);
		do_spi_check('h 99);
		for (i = 14; i < 256; i = i+7) begin
			do_spi_xfer(i);
			do_spi_check(i - 14);
		end
		do_spi_csel(0);

		do_spi_csel(2);
		do_spi_xfer('h 00);
		for (i = 0; i < 32; i = i+1) begin
			do_spi_xfer(i*8 + 4);
		end
		do_spi_csel(0);

		do_spi_csel(3);
		do_spi_xfer('h 00);
		for (i = 0; i < 32; i = i+1) begin
			do_spi_xfer(i*8 + 4);
		end
		do_spi_csel(0);

		for (j = 10; j < 256; j=j+1) begin
			do_spi_csel(1);
			do_spi_xfer('h 0);
			do_spi_check(143);
			do_spi_xfer('h 1);
			do_spi_check(42);
			for (i = 2; i < j; i = i+1) begin
				do_spi_xfer(i);
				do_spi_check(i - 2);
			end
			do_spi_csel(0);
		end
	end

	initial begin
		if ($test$plusargs("vcd")) begin
			$dumpfile("chip_tb.vcd");
			$dumpvars(0, chip_tb);
		end

		repeat (SIM_CYCLES) @(posedge clk);
		$display("DONE.");
		$finish;
	end
endmodule
