
module example(clk, rst, ctrl, step, out);

input clk, rst, ctrl;
input [3:0] step;
output reg [3:0] out;
reg [2:0] state;

always @(posedge clk) begin
	if (rst) begin
		out <= 0;
		state <= 3'b001;
	end else begin
		case (state)
			3'b001: begin
				if (ctrl)
					state <= 3'b010;
			end
			3'b010: begin
				state <= 3'b100;
				if (!ctrl) begin
					state <= 3'b001;
					out <= 0;
				end
			end
			3'b100: begin
				state <= 3'b001;
				if (ctrl)
					out <= out + step;
				else
					out <= out - step;
			end
			default: begin
				state <= 'bx;
				out <= 'bx;
			end
		endcase
	end
end

endmodule

