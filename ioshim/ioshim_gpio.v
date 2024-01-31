module ioshim_gpio (
	input clk, resetn,

	input io_en,
	input [7:0] io_dout1,
	input [7:0] io_dout2,
	input [15:0] io_ab_dout,
	output reg io_wreg, io_wa, io_wb,
	output reg [7:0] io_din,
	output reg [15:0] io_ab_din,

	output reg [7:0] gpio_dir,
	output reg [7:0] gpio_dout,
	input [7:0] gpio_din
);
	always @(posedge clk) begin
		io_wreg <= 0;
		io_wa <= 0;
		io_wb <= 0;

		io_din <= 'bx;
		io_ab_din <= 'bx;

		if (!resetn) begin
			gpio_dir <= 0;
			gpio_dout <= 0;
		end else
		if (io_en) begin
			gpio_dir <= io_dout1;
			gpio_dout <= io_dout2;
			io_din <= gpio_din;
			io_wreg <= 1;
		end
	end
endmodule
