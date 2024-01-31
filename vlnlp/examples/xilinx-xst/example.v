
module example(clk, rst, in, out);

input clk, rst;
input [7:0] in;
output reg [7:0] out;

(* fsm_encoding="one-hot", safe_implementation="no" *)
integer state;

always @(posedge clk) begin
	out <= 0;
	if (rst) begin
		state <= 100;
	end else begin
		case (state)
			100: begin
				out <= "H";
				state <= 101;
			end
			101: begin
				out <= "e";
				state <= 102;
			end
			102: begin
				out <= "l";
				state <= 103;
			end
			103: begin
				out <= "l";
				state <= 104;
			end
			104: begin
				out <= "o";
				state <= 105;
			end
			105: begin
				out <= "!";
				state <= 106;
			end
			106: begin
				out <= "\n";
				state <= 200;
			end
			200: begin
				if (in == "\n")
					state <= 100;
				out <= in;
			end
		endcase
	end
end

endmodule

