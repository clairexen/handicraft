module ioshim_cpu (
	input clk,
	input resetn,
	input [10:0] rstaddr,

	input memio_rd,
	input [1:0] memio_wr,
	input [10:0] memio_addr,
	input [15:0] memio_wdata,
	output [15:0] memio_rdata,
	output reg memio_done,

	output reg io_en,
	output reg [4:0] io_epnum,
	output reg [7:0] io_dout1,
	output reg [7:0] io_dout2,
	output reg [15:0] io_ab_dout,
	input io_wreg, io_wa, io_wb,
	input [7:0] io_din,
	input [15:0] io_ab_din
);
	parameter integer MEMSIZE = 128;
	parameter MEMINIT = "";

	reg [11:0] ticks;
	reg sync_continue;

	reg [10:0] pc;
	reg [10:0] next_pc;
	reg [10:0] branch_pc;
	reg [10:0] link_pc;
	reg branch_z, branch_nz;
	wire [15:0] opcode;

	reg [15:0] abreg;
	reg write_ab_from_pc;
	reg write_a_from_reg;
	reg write_b_from_reg;
	reg [1:0] write_ab_from_mem;

	reg write_reg;
	reg write_reg_from_alu;
	reg write_reg_from_imm;
	reg write_reg_from_io;
	reg write_reg_from_a;
	reg write_reg_from_b;
	reg [1:0] write_reg_from_mem;
	reg [3:0] write_reg_idx;
	reg [7:0] write_reg_value;
	wire [7:0] regval1, regval2;
	reg written_zero;

	wire [7:0] alu_result;
	reg [7:0] imm_buffer;
	reg [7:0] regval1_buffer;

	reg [1:0] mem_wen;
	reg [10:0] mem_addr;
	reg [15:0] mem_wdata;
	wire [15:0] mem_rdata;
	reg mem_memio;

	always @(posedge clk)
		regval1_buffer <= regval1;

	always @* begin
		casez (opcode)
			16'b 1111_1000_zzzz_zzzz: begin // ld
				mem_wen = 0;
				mem_addr = regval2 >> 1;
				mem_wdata = 'bx;
				mem_memio = 0;
			end
			16'b 1111_1001_zzzz_zzzz: begin // st
				mem_wen = regval2[0] ? 2 : 1;
				mem_addr = regval2 >> 1;
				mem_wdata = {2{regval1}};
				mem_memio = 0;
			end
			16'b 1111_1010_zzzz_zzzz: begin // ldab
				mem_wen = 0;
				mem_addr = {regval1, regval2} >> 1;
				mem_wdata = 'bx;
				mem_memio = 0;
			end
			16'b 1111_1011_zzzz_zzzz: begin // stab
				mem_wen = 3;
				mem_addr = {regval1, regval2} >> 1;
				mem_wdata = regval2[0] ? {abreg[7:0], abreg[15:8]} : abreg;
				mem_memio = 0;
			end
			default: begin
				mem_wen = {2{memio_wr}};
				mem_addr = memio_addr;
				mem_wdata = memio_wdata;
				mem_memio = memio_rd || memio_wr;
			end
		endcase
	end

	always @(posedge clk)
		memio_done <= mem_memio && !memio_done;

	assign memio_rdata = mem_rdata;

	ioshim_regs regs (
		.clk(clk),
		.resetn(resetn),

		.wr_en(write_reg),
		.wr_addr(write_reg_idx),
		.wr_data(write_reg_value),

		.rd1_addr(opcode[15] ? opcode[7:4] : {1'b0, opcode[14:13], opcode[7]}),
		.rd1_data(regval1),

		.rd2_addr(opcode[3:0]),
		.rd2_data(regval2)
	);

	ioshim_mem #(
		.MEMSIZE(MEMSIZE),
		.MEMINIT(MEMINIT)
	) mem (
		.clk(clk),
		.resetn(resetn),

		.wen1(mem_wen),
		.addr1(mem_addr),
		.wdata1(mem_wdata),
		.rdata1(mem_rdata),

		.addr2(next_pc),
		.rdata2(opcode)
	);

	ioshim_alu alu (
		.clk   (clk         ),
		.resetn(resetn      ),
		.insn  (opcode[12:8]),
		.op1   (regval1     ),
		.op2   (regval2     ),
		.result(alu_result  )
	);

	always @* begin
		if (!resetn) begin
			next_pc = 0;
		end else if (opcode[15:12] == 4'b1101 && !sync_continue) begin
			next_pc = pc;
		end else if (written_zero ? branch_z : branch_nz) begin
			next_pc = branch_pc;
		end else begin
			next_pc = pc + 1;
		end
	end

	always @* begin
		write_reg_value = 1'bx;
		write_reg = |{write_reg_from_alu, write_reg_from_imm, write_reg_from_io && io_wreg, write_reg_from_mem, write_reg_from_a, write_reg_from_b};

		(* parallel_case *)
		case (1)
			write_reg_from_alu:    write_reg_value = alu_result;
			write_reg_from_imm:    write_reg_value = imm_buffer;
			write_reg_from_io:     write_reg_value = io_din;
			write_reg_from_mem[0]: write_reg_value = mem_rdata[ 7:0];
			write_reg_from_mem[1]: write_reg_value = mem_rdata[15:8];
			write_reg_from_a:      write_reg_value = abreg[15:8];
			write_reg_from_b:      write_reg_value = abreg[7:0];
		endcase
	end

	always @* begin
		io_en = opcode[15:13] == 3'b101;
		io_epnum = io_en ? opcode[12:8] : 'bx;
		io_dout1 = io_en ? regval1 : 'bx;
		io_dout2 = io_en ? regval2 : 'bx;
		io_ab_dout = abreg;
	end

	always @(posedge clk) begin
		(* parallel_case *)
		case (1)
			write_reg_from_io && io_wa: abreg[15:8] <= io_ab_din[15:8];
			write_reg_from_io && io_wb: abreg[7:0] <= io_ab_din[7:0];
			write_ab_from_mem[0]: abreg <= mem_rdata;
			write_ab_from_mem[1]: abreg <= {mem_rdata[7:0], mem_rdata[15:8]};
			write_ab_from_pc: abreg <= link_pc;
			write_a_from_reg: abreg[15:8] <= regval1_buffer;
			write_b_from_reg: abreg[7:0] <= regval1_buffer;
		endcase
	end

	always @(posedge clk) begin
		write_reg_from_alu <= 0;
		write_reg_from_imm <= 0;
		write_reg_from_io <= 0;
		write_reg_from_mem <= 0;
		write_ab_from_mem <= 0;
		write_ab_from_pc <= 0;
		write_a_from_reg <= 0;
		write_b_from_reg <= 0;
		write_reg_from_a <= 0;
		write_reg_from_b <= 0;
		write_reg_idx <= {opcode[7] & opcode[15], opcode[6:4]};
		imm_buffer <= {opcode[11:8], opcode[3:0]};

		branch_z <= 0;
		branch_nz <= 0;
		branch_pc <= 'bx;
		written_zero <= write_reg && !write_reg_value;
		link_pc <= pc + 1;
		pc <= next_pc;

		ticks <= ticks + |{~ticks};
		sync_continue <= 0;

		if (!resetn) begin
			pc <= rstaddr;
			ticks <= 0;
		end else begin
			casez (opcode)
				16'b 0zzz_zzzz_zzzz_zzzz: begin // ALU 3-Op
					write_reg_from_alu <= 1;
				end
				16'b 100z_zzzz_1zzz_zzzz: begin // ALU 2-Op
					write_reg_from_alu <= 1;
				end
				16'b 101z_zzzz_zzzz_zzzz: begin // IO Op
					write_reg_from_io <= 1;
				end
				16'b 1100_zzzz_zzzz_zzzz: begin // Load Imm
					write_reg_from_imm <= 1;
				end
				16'b 1101_zzzz_zzzz_zzzz: begin // Sync
					if (ticks >= opcode[11:0] && !sync_continue) begin
						ticks <= ticks - opcode[11:0];
						sync_continue <= 1;
					end
				end
				16'b 1110_0zzz_zzzz_zzzz: begin // Branch
					branch_pc <= opcode[10:0];
					branch_z <= 1;
					branch_nz <= 1;
				end
				16'b 1110_1zzz_zzzz_zzzz: begin // Branch Z
					branch_pc <= opcode[10:0];
					branch_z <= 1;
				end
				16'b 1111_0zzz_zzzz_zzzz: begin // Branch NZ
					branch_pc <= opcode[10:0];
					branch_nz <= 1;
				end
				16'b 1111_1000_zzzz_zzzz: begin // ld
					write_reg_from_mem <= regval2[0] ? 2 : 1;
				end
				16'b 1111_1001_zzzz_zzzz: begin // st
					/* nothing to do here */
				end
				16'b 1111_1010_zzzz_zzzz: begin // ldab
					write_ab_from_mem <= regval2[0] ? 2 : 1;
				end
				16'b 1111_1011_zzzz_zzzz: begin // stab
					/* nothing to do here */
				end
				16'b 1111_1100_zzzz_0000: begin // seta
					write_a_from_reg <= 1;
				end
				16'b 1111_1100_zzzz_0001: begin // setb
					write_b_from_reg <= 1;
				end
				16'b 1111_1100_zzzz_0010: begin // geta
					write_reg_from_a <= 1;
				end
				16'b 1111_1100_zzzz_0011: begin // getb
					write_reg_from_b <= 1;
				end
				16'b 1111_1111_0000_0000: begin // jab
					branch_pc <= abreg;
					branch_z <= 1;
					branch_nz <= 1;
				end
				16'b 1111_1111_0000_0001: begin // lab
					write_ab_from_pc <= 1;
				end
			endcase
		end
	end
endmodule
