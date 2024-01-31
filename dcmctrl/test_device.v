module top (
	input clk,

	input spi_ss,
	input spi_clk,
	input spi_mosi,
	output spi_miso,

	output debug_ss,
	output debug_clk,
	output debug_mosi,
	output debug_miso,

	output pwm_left,
	output pwm_right,
	output pwm_pulse
);
	reg [15:0] pulse_sim_cnt = 0;

	always @(posedge clk)
		pulse_sim_cnt <= pulse_sim_cnt + 1;
	
	assign pwm_pulse = !pulse_sim_cnt && (pwm_left || pwm_right);

	reg [7:0] reset_cnt = 0;
	wire reset = &reset_cnt == 0;

	always @(posedge clk)
		reset_cnt <= reset_cnt + reset;

	dcmctrl #(
		.N_CHANNELS(1)
	) uut (
		.clk(clk),
		.reset(reset),

		.spi_ss     (spi_ss   ),
		.spi_clk    (spi_clk  ),
		.spi_mosi   (spi_mosi ),
		.spi_miso   (spi_miso ),

		.motor_left (pwm_left ),
		.motor_right(pwm_right),
		.motor_pulse(pwm_pulse),
		.motor_fault(    1'b 0),
		.motor_otw  (    1'b 0)
	);

	assign debug_ss   = spi_ss;
	assign debug_clk  = spi_clk;
	assign debug_mosi = spi_mosi;
	assign debug_miso = spi_miso;
endmodule
