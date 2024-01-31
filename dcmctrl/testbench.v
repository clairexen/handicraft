module testbench;
	reg clk = 1;
	always #5 clk = ~clk;

	reg reset;

	integer cycle;
	always @(posedge clk) cycle <= reset ? 0 : cycle + 1;

	reg spi_ss;
	reg spi_clk;
	reg spi_mosi;
	wire spi_miso;

	wire [7:0] motor_left;
	wire [7:0] motor_right;
	wire [7:0] motor_reset;

	reg  [7:0] motor_pulse;
	reg  [7:0] motor_fault;
	reg  [7:0] motor_otw;

	wire motor_left_0 = motor_left[0];
	wire motor_right_0 = motor_right[0];
	wire motor_reset_0 = motor_reset[0];
	wire motor_pulse_0 = motor_pulse[0];
	wire motor_fault_0 = motor_fault[0];
	wire motor_otw_0 = motor_otw[0];

	dcmctrl uut (
		.clk(clk),
		.reset(reset),
		.spi_ss(spi_ss),
		.spi_clk(spi_clk),
		.spi_mosi(spi_mosi),
		.spi_miso(spi_miso),
		.motor_left(motor_left),
		.motor_right(motor_right),
		.motor_reset(motor_reset),
		.motor_pulse(motor_pulse),
		.motor_fault(motor_fault),
		.motor_otw(motor_otw)
	);

	reg [7:0] spidata;

	task spi_xfer_begin;
		begin
			spi_ss <= 0;
		end
	endtask

	task spi_xfer_byte(input [7:0] d);
		integer i;
		begin
			spidata = d;
			for (i = 0; i < 8; i = i+1) begin
				spi_mosi <= spidata[7];
				#50;
				spi_clk <= 1;
				spidata = {spidata, spi_miso};
				#50;
				spi_clk <= 0;
			end

			// add extra delay after each byte for
			// improved readability of VCD traces
			#100;
		end
	endtask

	task spi_xfer_end;
		begin
			spi_ss <= 1;
			#100;
		end
	endtask

	integer i;

	initial begin
		$dumpfile("testbench.vcd");
		$dumpvars(0, testbench);

		spi_ss <= 1;
		spi_clk <= 0;
		spi_mosi <= 0;
		reset <= 1;

		motor_pulse <= 0;
		motor_fault <= 0;
		motor_otw <= 0;

		#10000;

		repeat (100) @(posedge clk);
		reset <= 0;

		// Zero initialize register file and reset

		spi_xfer_begin;
		spi_xfer_byte(128);
		for (i = 0; i < 128; i = i+1)
			spi_xfer_byte(0);
		spi_xfer_end;

		repeat (10) @(posedge clk);
		reset <= 1;

		repeat (10) @(posedge clk);
		reset <= 0;

		// Set target speed and position for channel 0

		spi_xfer_begin;
		spi_xfer_byte(128 | 64);
		spi_xfer_byte(100); // target speed
		spi_xfer_byte(0); // target position [23:16]
		spi_xfer_byte(0); // target position [15:8]
		spi_xfer_byte(80); // target position [7:0]
		spi_xfer_end;

		#500000;

		// motor_fault[0] <= 1;
		// motor_otw[0] <= 1;

		#1000000;

		// Read target speed and position for channel 0

		spi_xfer_begin;
		spi_xfer_byte(64);
		spi_xfer_byte(0);
		$display("target speed: %d", spidata);
		spi_xfer_byte(0);
		$display("target position [23:16]: %d", spidata);
		spi_xfer_byte(0);
		$display("target position [15: 8]: %d", spidata);
		spi_xfer_byte(0);
		$display("target position [ 7: 0]: %d", spidata);
		spi_xfer_end;

		// Read flags and current position for channel 0

		spi_xfer_begin;
		spi_xfer_byte(0);
		spi_xfer_byte(0);
		$display("flags: %d", spidata);
		spi_xfer_byte(0);
		$display("current position [23:16]: %d", spidata);
		spi_xfer_byte(0);
		$display("current position [15: 8]: %d", spidata);
		spi_xfer_byte(0);
		$display("current position [ 7: 0]: %d", spidata);
		spi_xfer_end;

		$finish;
	end

	initial begin
		motor_pulse <= 0;
		#500;

		forever begin
			motor_pulse <= motor_left | motor_right;
			#123;
			motor_pulse <= 0;
			#3777;
		end
	end
endmodule
