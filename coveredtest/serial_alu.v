module serial_alu (
	input        clock,
	input        reset,
	input  [7:0] din_di1,
	input  [7:0] din_di2,
	input  [1:0] din_fun,
	input        din_vld,
	output       din_rdy,
	output [7:0] dout_dat,
	output       dout_vld
);
	reg [3:0] state;
	reg [7:0] data1;
	reg [7:0] data2;
	reg [1:0] func;
	reg carry, busy;

	assign din_rdy = (state == 0);
	assign dout_dat = data1;
	assign dout_vld = busy && din_rdy;

	always @(posedge clock) begin
		if (reset || state == 0) begin
			state <= (!reset && din_vld) << 3;
			busy <= (!reset && din_vld);
			carry <= 0;
			data1 <= din_di1;
			data2 <= din_di2;
			func <= din_fun;
		end else begin
			state <= state - 1;
			busy <= 1;
			carry <= (data1[0] && data2[0]) || (data1[0] && carry) || (data2[0] && carry);
			data1 <= data1 >> 1;
			data2 <= data2 >> 1;
			case (func)
				0: data1[7] <= data1[0] ^ data2[0] ^ carry;
				1: data1[7] <= data1[0] & data2[0];
				2: data1[7] <= data1[0] | data2[0];
				3: data1[7] <= data1[0] == data2[0];   // BUG (not caught by testbench)
			endcase
		end
	end
endmodule
