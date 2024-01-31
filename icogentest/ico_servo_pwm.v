module ico_servo_pwm #(
	parameter integer NUM_PMODS = 1,
	parameter integer CLK_KHZ = 12000
) (
	input clk,
	input resetn,
	input spi_ctrl_si,
	input spi_ctrl_so,
	input spi_ctrl_hd,
	input [7:0] spi_ctrl_di,
	output [7:0] spi_ctrl_do,
	input [1:0] epsel,
	input [8*NUM_PMODS-1:0] pmod_i,
	output [8*NUM_PMODS-1:0] pmod_o,
	output [8*NUM_PMODS-1:0] pmod_d
);
	reg [15:0] config_mem [0:8*NUM_PMODS-1];
	reg [7:0] cfg_addr;
	reg [1:0] cfg_state;

	always @(posedge clk) begin
		if (!resetn) begin
			cfg_state <= 0;
		end else
		case (cfg_state)
			0: begin
				cfg_addr <= spi_ctrl_di;
				if (spi_ctrl_hd && spi_ctrl_si) begin
					if (epsel[0]) cfg_state <= 1;
					if (epsel[1]) cfg_state <= 2;
				end
			end
			1: begin
				if (!epsel[0]) begin
					cfg_state <= 0;
				end else
				if (spi_ctrl_si) begin
					config_mem[cfg_addr][7:0] <= spi_ctrl_di;
					cfg_addr <= cfg_addr + 1;
				end
			end
			2: begin
				if (!epsel[1]) begin
					cfg_state <= 0;
				end else
				if (spi_ctrl_si) begin
					config_mem[cfg_addr][15:8] <= spi_ctrl_di;
					cfg_addr <= cfg_addr + 1;
				end
			end
		endcase
	end

	assign spi_ctrl_do = 0;

	reg [8*NUM_PMODS-1:0] pins, pins_sr;
	assign pmod_d = ~0, pmod_o = pins;

	reg [7:0] current_addr = 0, next_addr;
	reg [10:0] timer_10us = 0, this_start, this_stop;
	reg [15:0] current_cfg;

	always @(posedge clk) begin
		next_addr = current_addr == (CLK_KHZ/100 - 1) ? 0 : current_addr + 1;
		current_cfg <= config_mem[next_addr];
		current_addr <= next_addr;

		if (current_addr == 0) begin
			timer_10us <= timer_10us + 1;
			pins <= pins_sr;
		end

		if (current_addr < 8*NUM_PMODS) begin
			this_start = current_cfg[15:8] << 3;
			this_stop = (current_cfg[15:8] << 3) + current_cfg[7:0];

			if (timer_10us == this_stop)
				pins_sr <= {1'b0, pins_sr[8*NUM_PMODS-1:1]};
			else if (timer_10us == this_start)
				pins_sr <= {1'b1, pins_sr[8*NUM_PMODS-1:1]};
			else
				pins_sr <= {pins_sr[0], pins_sr[8*NUM_PMODS-1:1]};
		end
	end
endmodule
