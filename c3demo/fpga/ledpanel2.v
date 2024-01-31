// Description of the LED panel:
// http://bikerglen.com/projects/lighting/led-panel-1up/#The_LED_Panel
//
// PANEL_[ABCD] ... select rows (in pairs from top and bottom half)
// PANEL_OE ....... display the selected rows (active low)
// PANEL_CLK ...... serial clock for color data
// PANEL_STB ...... latch shifted data (active high)
// PANEL_[RGB]0 ... color channel for top half
// PANEL_[RGB]1 ... color channel for bottom half

module ledpanel (
	input clk,

	input wr_enable,
	input [4:0] wr_addr_x,
	input [4:0] wr_addr_y,
	input [23:0] wr_rgb_data,

	output reg PANEL_R0, PANEL_G0, PANEL_B0, PANEL_R1, PANEL_G1, PANEL_B1,
	output reg PANEL_A, PANEL_B, PANEL_C, PANEL_D, PANEL_CLK, PANEL_STB, PANEL_OE
);
	reg [7:0] video_mem_r [0:1023];
	reg [7:0] video_mem_g [0:1023];
	reg [7:0] video_mem_b [0:1023];

	always @(posedge clk) begin
		if (wr_enable) begin
			video_mem_r[{wr_addr_x, wr_addr_y}] <= wr_rgb_data[23:16];
			video_mem_g[{wr_addr_x, wr_addr_y}] <= wr_rgb_data[15: 8];
			video_mem_b[{wr_addr_x, wr_addr_y}] <= wr_rgb_data[ 7: 0];
		end
	end

	reg [8:0] cnt_x = 0;
	reg [3:0] cnt_y = 0;
	reg [2:0] cnt_z = 0;
	reg state = 0;

	reg [4:0] addr_x;
	reg [4:0] addr_y;
	reg [2:0] addr_z;
	reg [2:0] data_rgb;
	reg [2:0] data_rgb_q;
	reg [8:0] max_cnt_x;

	always @(posedge clk) begin
		case (cnt_z)
			0: max_cnt_x <= 4*32 + 2;
			1: max_cnt_x <= 4*32 + 2;
			2: max_cnt_x <= 4*32 + 2;
			3: max_cnt_x <= 4*32 + 2;
			4: max_cnt_x <= 4*32 + 2;
			5: max_cnt_x <= 4*32 + 2;
			6: max_cnt_x <= 4*32 + 2;
			7: max_cnt_x <= 256;
		endcase
	end

	always @(posedge clk) begin
		state <= !state;
		if (!state) begin
			if (cnt_x > max_cnt_x) begin
				cnt_x <= 0;
				cnt_z <= cnt_z + 1;
				if (&cnt_z)
					cnt_y <= cnt_y + 1;
			end else begin
				cnt_x <= cnt_x + 1;
			end
		end
	end

	always @(posedge clk) begin
		PANEL_OE <= cnt_z == 0 ||
				(cnt_z == 1 && cnt_x > 1) ||
				(cnt_z == 2 && cnt_x > 3) ||
				(cnt_z == 3 && cnt_x > 7) ||
				(cnt_z == 4 && cnt_x > 15);
		if (state) begin
			PANEL_CLK <= 1 < cnt_x && cnt_x < (4*32+2);
			PANEL_STB <= cnt_x == (4*32+2);
		end else begin
			PANEL_CLK <= 0;
			PANEL_STB <= 0;
		end
	end

	always @(posedge clk) begin
		addr_x <= cnt_x[5:1];
		addr_y <= {!cnt_x[6], !state, cnt_y[3:1]};
		addr_z <= cnt_z;
	end

	always @(posedge clk) begin
		data_rgb[2] <= video_mem_r[{addr_x, addr_y}][addr_z];
		data_rgb[1] <= video_mem_b[{addr_x, addr_y}][addr_z];
		data_rgb[0] <= video_mem_g[{addr_x, addr_y}][addr_z];
	end

	always @(posedge clk) begin
		data_rgb_q <= data_rgb;
		if (!state) begin
			{PANEL_R1, PANEL_R0} <= {data_rgb[2], data_rgb_q[2]};
			{PANEL_G1, PANEL_G0} <= {data_rgb[1], data_rgb_q[1]};
			{PANEL_B1, PANEL_B0} <= {data_rgb[0], data_rgb_q[0]};
		end
		if (PANEL_STB) begin
			{PANEL_D, PANEL_C, PANEL_B, PANEL_A} <= cnt_y;
		end
	end

/*
	integer oe_cnt = 0;
	always @(posedge clk) begin
		if (PANEL_OE || PANEL_STB) begin
			if (oe_cnt)
				$display("OE Cycles: %3d", oe_cnt);
			oe_cnt <= 0;
		end else
			oe_cnt <= oe_cnt + 1;
	end
*/
endmodule
