module ico_echo_test #(
	parameter integer NUM_PMODS = 0,
	parameter integer CLK_KHZ = 12000
) (
	input clk,
	input resetn,
	input spi_ctrl_si,
	input spi_ctrl_so,
	input spi_ctrl_hd,
	input [7:0] spi_ctrl_di,
	output [7:0] spi_ctrl_do,
	input epsel
);
	reg [7:0] last_word;

	always @(posedge clk) begin
		if (!resetn) begin
			last_word <= 99; // never user
		end else begin
			if (spi_ctrl_si)
				last_word <= spi_ctrl_di; // echo back (starting with third byte)
			else if (spi_ctrl_hd)
				last_word <= 143; // first out byte
			else if (spi_ctrl_so)
				last_word <= 42; // second out byte
		end
	end

	assign spi_ctrl_do = epsel ? last_word : 0;
endmodule
