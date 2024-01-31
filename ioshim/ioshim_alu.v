module ioshim_alu (
	input clk,
	input resetn,

	input [4:0] insn,
	input [7:0] op1, 
	input [7:0] op2, 
	output reg [7:0] result
);
	integer i, j;
	reg [7:0] packed_result;
	reg [7:0] unpacked_result;

	always @(op1, op2) begin
		packed_result = op1;
		for (i = 7; i >= 0; i = i-1) begin
			if (!op2[i]) begin
				for (j = i; j < 8; j = j+1)
					packed_result[j] = packed_result >> (j+1);
			end
		end

		unpacked_result = op1;
		for (i = 0; i < 8; i = i+1) begin
			if (!op2[i]) begin
				for (j = 7; j > i; j = j-1)
					unpacked_result[j] = op1 >> (j-1);
				unpacked_result[i] = 1'b0;
			end
		end
	end

	always @(posedge clk) begin
		result <= 'bx;

		case (insn)
		 0: result <= op2;
		 1: result <= op1 + op2;
		 2: result <= op1 - op2;
		 3: result <= op1 << op2;
		 4: result <= op1 >> op2;
		 5: result <= $signed(op1) >>> op2;
		 6: result <= packed_result;
		 7: result <= unpacked_result;
		 8: result <= op1 <  op2;
		 9: result <= op1 <= op2;
		10: result <= op1 != op2;
		11: result <= op1 == op2;
		12: result <= op1 >= op2;
		13: result <= op1 >  op2;
		14: result <= $signed(op1) <  $signed(op2);
		15: result <= $signed(op1) <= $signed(op2);
		16: result <= $signed(op1) >= $signed(op2);
		17: result <= $signed(op1) >  $signed(op2);
		18: result <= ({1'b0, op1} + {1'b0, op2}) >> 8;
		19: result <= ({1'b0, op1} - {1'b0, op2}) >> 8;
		20: result <= !op2;
		21: result <= |op2;
		22: result <= ~op2;
		23: result <= op1 & op2;
		24: result <= op1 | op2;
		25: result <= op1 ^ op2;
		26: result <= {op2[0], op2[1], op2[2], op2[3], op2[4], op2[5], op2[6], op2[7]};
		endcase
	end
endmodule
