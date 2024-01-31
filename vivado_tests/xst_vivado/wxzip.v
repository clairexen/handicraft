module wxzip(clk, rst, din, dout, wr_en, rd_en, rd_empty, error, half_full);

// Only one of this parameters may be set to 1
parameter MODE_2_10 = 0;
parameter MODE_4_12 = 1;

localparam selected_mode =
	MODE_2_10 == 1 && MODE_4_12 == 0 ? 210 :
	MODE_2_10 == 0 && MODE_4_12 == 1 ? 412 : 0;

input clk, rst;
input [33:0] din;
output reg [38:0] dout;
input wr_en, rd_en;
output reg rd_empty, error, half_full;

parameter BUFFER_PTR_BITS = 9;
localparam BUFFER_SZ = 2 ** BUFFER_PTR_BITS;
localparam [BUFFER_PTR_BITS-1:0] BUFFER_HALF_SZ = 2 ** (BUFFER_PTR_BITS-1);

reg [33:0] buffer_in [0:BUFFER_SZ-1];
reg [38:0] buffer_out [0:BUFFER_SZ-1];
reg [33:0] buffer_in_tmp;
reg buffer_in_ready;

reg [BUFFER_PTR_BITS-1:0] buffer_in_ptr_wr;
reg [BUFFER_PTR_BITS-1:0] buffer_in_ptr_rd_engine;
reg [BUFFER_PTR_BITS-1:0] buffer_in_ptr_rd_output;
reg [BUFFER_PTR_BITS-1:0] buffer_out_ptr_wr;
reg [BUFFER_PTR_BITS-1:0] buffer_out_ptr_rd;
reg [BUFFER_PTR_BITS-1:0] next_ptr;

reg output_from_buffer_in;
reg output_from_buffer_out;

(* fsm_encoding = "one-hot" *)
integer engine_state;

reg [31:0] engine_in_bytes;
reg engine_finish_flag;
reg [1:0] engine_in_bytes_count;
reg [7:0] engine_last_sample;
reg [11:0] engine_push_data;
reg engine_push_2bits;
reg engine_push_4bits;
reg engine_push_10bits;
reg engine_push_12bits;
reg engine_push_flush;
reg [31:0] engine_push_buffer;
reg [3:0] engine_push_count;
reg [4:0] engine_padding_bits;
reg [BUFFER_PTR_BITS-1:0] engine_in_count;
reg [BUFFER_PTR_BITS-1:0] engine_out_count;
reg signed [7:0] engine_delta;

task push_two_bits;
  input [1:0] bits;
  input end_marker;
  input decrement_padding;
  begin
	engine_push_buffer = { bits, engine_push_buffer[31:2] };
	engine_push_count = engine_push_count + 1;
	if (decrement_padding)
		engine_padding_bits = engine_padding_bits - 2;
	if (engine_push_count == 0) begin
		buffer_out[buffer_out_ptr_wr] <= { engine_padding_bits, end_marker ? 2'b11 : engine_out_count == 0 ? 2'b10 : 2'b00, engine_push_buffer };
		buffer_out_ptr_wr <= buffer_out_ptr_wr + 1;
		engine_out_count <= engine_out_count + 1;
		if (end_marker)
			engine_push_flush <= 0;
		engine_padding_bits = 0;
	end
  end
endtask

task push_four_bits;
  input [3:0] bits;
  input end_marker;
  input decrement_padding;
  begin
	engine_push_buffer = { bits, engine_push_buffer[31:4] };
	engine_push_count = engine_push_count + 2;
	if (decrement_padding)
		engine_padding_bits = engine_padding_bits - 4;
	if (engine_push_count == 0) begin
		buffer_out[buffer_out_ptr_wr] <= { engine_padding_bits, end_marker ? 2'b11 : engine_out_count == 0 ? 2'b10 : 2'b00, engine_push_buffer };
		buffer_out_ptr_wr <= buffer_out_ptr_wr + 1;
		engine_out_count <= engine_out_count + 1;
		if (end_marker)
			engine_push_flush <= 0;
		engine_padding_bits = 0;
	end
  end
endtask

always @(posedge clk)
  begin
	half_full <= 0;
	if (buffer_in_ptr_wr - buffer_in_ptr_rd_engine >= BUFFER_HALF_SZ)
		half_full <= 1;
	if (buffer_in_ptr_wr - buffer_in_ptr_rd_output >=  BUFFER_HALF_SZ)
		half_full <= 1;
	if (buffer_out_ptr_wr - buffer_out_ptr_rd >= BUFFER_HALF_SZ)
		half_full <= 1;

	engine_push_data <= 'bx;
	engine_push_2bits <= 0;
	engine_push_4bits <= 0;
	engine_push_10bits <= 0;
	engine_push_12bits <= 0;
	engine_delta = 'bx;

	buffer_in_tmp <= buffer_in[buffer_in_ptr_rd_engine];
	buffer_in_ready <= buffer_in_ptr_wr != buffer_in_ptr_rd_engine;

	if (rst)
	  begin
		buffer_in_ptr_wr <= 0;
		buffer_in_ptr_rd_engine <= 0;
		buffer_in_ptr_rd_output <= 0;

		buffer_out_ptr_wr <= 0;
		buffer_out_ptr_rd <= 0;

		error <= 0;
		output_from_buffer_in <= 0;
		output_from_buffer_out <= 0;

		engine_state <= 100;
		engine_in_bytes <= 'bx;
		engine_finish_flag <= 'bx;
		engine_in_bytes_count <= 0;
		engine_last_sample <= 0;
		engine_push_flush <= 0;
		engine_push_buffer = 'bx;
		engine_push_count = 0;
		engine_padding_bits = 0;
		engine_in_count <= 0;
		engine_out_count <= 0;
	  end
	else
	  begin
	  	if (wr_en) begin
			next_ptr = buffer_in_ptr_wr + 1;
			if (next_ptr == buffer_in_ptr_rd_engine || next_ptr == buffer_in_ptr_rd_output)
				error <= 1;
			buffer_in[buffer_in_ptr_wr] <= din;
			buffer_in_ptr_wr <= next_ptr;
		end

		if (rd_en && !rd_empty) begin
			if (output_from_buffer_in) begin
				if (buffer_in[buffer_in_ptr_rd_output][33:32] == 2'b11)
					output_from_buffer_in <= 0;
				dout <= { 5'd0, buffer_in[buffer_in_ptr_rd_output] };
				buffer_in_ptr_rd_output <= buffer_in_ptr_rd_output + 1;
			end else
			if (output_from_buffer_out) begin
				if (buffer_out[buffer_out_ptr_rd][33:32] == 2'b11)
					output_from_buffer_out <= 0;
				dout <= buffer_out[buffer_out_ptr_rd];
				buffer_out_ptr_rd <= buffer_out_ptr_rd + 1;
			end else
				error <= 1;
		end

		case (engine_state)
			100: begin
				if (buffer_in_ready) begin
					engine_finish_flag <= buffer_in_tmp[33:32] == 2'b11;
					engine_in_bytes <= buffer_in_tmp[31:0];
					buffer_in_ptr_rd_engine <= buffer_in_ptr_rd_engine + 1;
					engine_in_count <= engine_in_count + 1;
					engine_state <= 200;
				end
			  end
			200: begin
				engine_delta = engine_in_bytes[7:0] - engine_last_sample;
				if (selected_mode == 210) begin
					if (-1 <= engine_delta && engine_delta <= +1) begin
						engine_push_data <= { 10'bx, engine_delta[1:0] };
						engine_push_2bits <= 1;
					end else begin
						engine_push_data <= { 2'bx, engine_in_bytes[7:0], 2'b10 };
						engine_push_10bits <= 1;
					end
				end
				if (selected_mode == 412) begin
					if (-7 <= engine_delta && engine_delta <= +7) begin
						engine_push_data <= { 8'bx, engine_delta[3:0] };
						engine_push_4bits <= 1;
					end else begin
						engine_push_data <= { engine_in_bytes[7:0], 4'b1000 };
						engine_push_12bits <= 1;
					end
				end
				if (engine_in_bytes_count == 3) begin
					engine_state <= engine_finish_flag ? 300 : 100;
					engine_push_flush <= engine_finish_flag;
				end
				engine_last_sample <= engine_in_bytes[7:0];
				engine_in_bytes <= { 8'bx, engine_in_bytes[31:8] };
				engine_in_bytes_count <= engine_in_bytes_count + 1;
			  end
			300: begin
				if (!output_from_buffer_in && !output_from_buffer_out && !engine_push_flush) begin
					if (engine_in_count <= engine_out_count) begin
						output_from_buffer_in <= 1;
						buffer_out_ptr_rd <= buffer_out_ptr_wr;
					end else begin
						output_from_buffer_out <= 1;
						buffer_in_ptr_rd_output <= buffer_in_ptr_rd_engine;
					end
					engine_in_count <= 0;
					engine_out_count <= 0;
					engine_last_sample <= 0;
					engine_push_count = 0;
					engine_state <= 100;
				end
			  end
		endcase

		if (selected_mode == 210) begin
			if (engine_push_2bits) begin
				push_two_bits(engine_push_data[1:0], engine_push_flush, engine_push_2bits);
			end else
			if (engine_push_10bits || engine_push_flush) begin
				push_two_bits(engine_push_10bits ? engine_push_data[1:0] : 2'b00, !engine_push_10bits, engine_push_10bits);
				push_two_bits(engine_push_10bits ? engine_push_data[3:2] : 2'b00, !engine_push_10bits, engine_push_10bits);
				push_two_bits(engine_push_10bits ? engine_push_data[5:4] : 2'b00, !engine_push_10bits, engine_push_10bits);
				push_two_bits(engine_push_10bits ? engine_push_data[7:6] : 2'b00, !engine_push_10bits, engine_push_10bits);
				push_two_bits(engine_push_10bits ? engine_push_data[9:8] : 2'b00, engine_push_flush,   engine_push_10bits);
			end
		end

		if (selected_mode == 412) begin
			if (engine_push_4bits) begin
				push_four_bits(engine_push_data[3:0], engine_push_flush, engine_push_4bits);
			end else
			if (engine_push_12bits || engine_push_flush) begin
				push_four_bits(engine_push_12bits ? engine_push_data[ 3:0] : 4'b0000, !engine_push_12bits, engine_push_12bits);
				push_four_bits(engine_push_12bits ? engine_push_data[ 7:4] : 4'b0000, !engine_push_12bits, engine_push_12bits);
				push_four_bits(engine_push_12bits ? engine_push_data[11:8] : 4'b0000, engine_push_flush,   engine_push_12bits);
			end
		end
	end

	// overwrite output if no valid mode selected
	if (selected_mode == 0) begin
		dout <= 'bx;
		error <= 1;
	end
end

always @* begin
	rd_empty <= 1;
	if (!rst && output_from_buffer_in && buffer_in_ptr_rd_output != buffer_in_ptr_wr)
		rd_empty <= 0;
	if (!rst && output_from_buffer_out && buffer_out_ptr_rd != buffer_out_ptr_wr)
		rd_empty <= 0;
end

endmodule
