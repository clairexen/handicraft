module tinycpu (
	input clk,
	input resetn,

	output reg [3:0] op_strb,
	output reg [7:0] op_dout,
	input      [7:0] op_din,

	output reg [ 7:0] insn_addr,
	input      [11:0] insn_opcode
);
	reg [7:0] current_pc;
	reg [7:0] registers [0:15];

	reg [3:0] alu_out_regidx;
	reg [7:0] alu_out_value;
	reg       alu_out_enable;

	reg skip_next_insn;
	reg skip_current_insn;

	always @* begin
		op_strb = 0;
		op_dout = 'bx;

		insn_addr = current_pc + 1;
		skip_next_insn = 0;

		alu_out_regidx = insn_opcode[7:4];
		alu_out_value = 'bx;
		alu_out_enable = 0;

		if (!resetn) begin
			insn_addr = 0;
		end else
		if (!skip_current_insn) begin
			(* parallel_case, full_case *)
			casez (insn_opcode)
				12'b 0000_zzzz_zzzz: begin // mv
					alu_out_value = registers[insn_opcode[3:0]];
					alu_out_enable = 1;
				end
				12'b 0001_zzzz_zzzz: begin // xor
					alu_out_value = registers[insn_opcode[7:4]] ^ registers[insn_opcode[3:0]];
					alu_out_enable = 1;
				end
				12'b 0010_zzzz_zzzz: begin // and
					alu_out_value = registers[insn_opcode[7:4]] & registers[insn_opcode[3:0]];
					alu_out_enable = 1;
				end
				12'b 0011_zzzz_zzzz: begin // or
					alu_out_value = registers[insn_opcode[7:4]] | registers[insn_opcode[3:0]];
					alu_out_enable = 1;
				end
				12'b 0100_zzzz_zzzz: begin // add
					alu_out_value = registers[insn_opcode[7:4]] + registers[insn_opcode[3:0]];
					alu_out_enable = 1;
				end
				12'b 0101_zzzz_zzzz: begin // sub
					alu_out_value = registers[insn_opcode[7:4]] - registers[insn_opcode[3:0]];
					alu_out_enable = 1;
				end
				12'b 0110_zzzz_zzzz: begin // clt
					skip_next_insn = registers[insn_opcode[7:4]] < registers[insn_opcode[3:0]];
				end
				12'b 0111_zzzz_zzzz: begin // ceq
					skip_next_insn = registers[insn_opcode[7:4]] == registers[insn_opcode[3:0]];
				end
				12'b 1000_zzzz_zzzz: begin // li
					alu_out_value = $signed(insn_opcode[3:0]);
					alu_out_enable = 1;
				end
				12'b 1001_zzzz_zzzz: begin // lui
					alu_out_value = {insn_opcode[3:0], registers[insn_opcode[7:4]][3:0]};
					alu_out_enable = 1;
				end
				12'b 1010_zzzz_zzzz: begin // j
					insn_addr = insn_opcode[7:0];
				end
				12'b 1011_zzzz_zzzz: begin // jal
					alu_out_regidx = 0;
					alu_out_value = insn_addr;
					alu_out_enable = 1;
					insn_addr = insn_opcode[7:0];
				end
				12'b 1100_zzzz_zzzz: begin // reserved binary op
				end
				12'b 1101_zzzz_zzzz: begin // reserved binary op
				end
				12'b 1110_zzzz_zzzz: begin // reserved binary op
				end
				12'b 1111_0000_0000: begin // ret
					insn_addr = registers[0];
				end
				12'b 1111_0001_0000: begin // cnot
					skip_next_insn = 1;
				end
				12'b 1111_0010_0000: begin // reserved op
				end
				12'b 1111_0011_0000: begin // reserved op
				end
				12'b 1111_0100_0000: begin // reserved op
				end
				12'b 1111_0101_0000: begin // reserved op
				end
				12'b 1111_0110_0000: begin // reserved op
				end
				12'b 1111_0111_0000: begin // reserved op
				end
				12'b 1111_1000_0000: begin // reserved op
				end
				12'b 1111_1001_0000: begin // reserved op
				end
				12'b 1111_1010_0000: begin // reserved op
				end
				12'b 1111_1011_0000: begin // reserved op
				end
				12'b 1111_1100_0000: begin // reserved op
				end
				12'b 1111_1101_0000: begin // reserved op
				end
				12'b 1111_1110_0000: begin // reserved op
				end
				12'b 1111_1111_0000: begin // reserved op
				end
				12'b 1111_zzzz_0001: begin // not
					alu_out_value = ~registers[insn_opcode[7:4]];
					alu_out_enable = 1;
				end
				12'b 1111_zzzz_0010: begin // shr
					alu_out_value = registers[insn_opcode[7:4]] >> 1;
					alu_out_enable = 1;
				end
				12'b 1111_zzzz_0011: begin // shl
					alu_out_value = registers[insn_opcode[7:4]] << 1;
					alu_out_enable = 1;
				end
				12'b 1111_zzzz_0100: begin // reserved unary op
				end
				12'b 1111_zzzz_0101: begin // reserved unary op
				end
				12'b 1111_zzzz_0110: begin // reserved unary op
				end
				12'b 1111_zzzz_0111: begin // reserved unary op
				end
				12'b 1111_zzzz_1000: begin // reserved unary op
				end
				12'b 1111_zzzz_1001: begin // reserved unary op
				end
				12'b 1111_zzzz_1010: begin // reserved unary op
				end
				12'b 1111_zzzz_1011: begin // reserved unary op
				end
				12'b 1111_zzzz_1100: begin // op0
					op_strb[0] = 1;
					op_dout = registers[insn_opcode[7:4]];
					alu_out_value = op_din;
					alu_out_enable = 1;
				end
				12'b 1111_zzzz_1101: begin // op1
					op_strb[1] = 1;
					op_dout = registers[insn_opcode[7:4]];
					alu_out_value = op_din;
					alu_out_enable = 1;
				end
				12'b 1111_zzzz_1110: begin // op2
					op_strb[2] = 1;
					op_dout = registers[insn_opcode[7:4]];
					alu_out_value = op_din;
					alu_out_enable = 1;
				end
				12'b 1111_zzzz_1111: begin // op3
					op_strb[3] = 1;
					op_dout = registers[insn_opcode[7:4]];
					alu_out_value = op_din;
					alu_out_enable = 1;
				end
			endcase
		end
	end

	always @(posedge clk) begin
		if (alu_out_enable)
			registers[alu_out_regidx] <= alu_out_value;
		current_pc <= insn_addr;
		skip_current_insn <= skip_next_insn;
	end
endmodule
