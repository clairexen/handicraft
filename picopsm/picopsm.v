/*
 *  PicoPsm - A small programmable state machine
 *
 *  Copyright (C) 2014  Clifford Wolf <clifford@clifford.at>
 *  
 *  Permission to use, copy, modify, and/or distribute this software for any
 *  purpose with or without fee is hereby granted, provided that the above
 *  copyright notice and this permission notice appear in all copies.
 *  
 *  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 *  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 *  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 *  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 *  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 *  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 */


module picopsm (
	input clk, resetn,

	// AXI4-lite master memory interface

	output        mem_axi_awvalid,
	input         mem_axi_awready,
	output [15:0] mem_axi_awaddr,
	output [ 2:0] mem_axi_awprot,

	output        mem_axi_wvalid,
	input         mem_axi_wready,
	output [ 7:0] mem_axi_wdata,

	input         mem_axi_bvalid,
	output        mem_axi_bready,

	output        mem_axi_arvalid,
	input         mem_axi_arready,
	output [15:0] mem_axi_araddr,
	output [ 2:0] mem_axi_arprot,

	input         mem_axi_rvalid,
	output        mem_axi_rready,
	input  [ 7:0] mem_axi_rdata
);

	localparam state_bits   = 2;
	localparam state_decode = 2'b00;
	localparam state_arg1   = 2'b01;
	localparam state_arg2   = 2'b10;
	localparam state_exec   = 2'b11;

	reg [15:0] pc;
	reg [state_bits-1:0] state;

	reg carry;
	reg cmp_result;
	reg [7:0] acc;

	reg [1:0] op_ind, op_ind2;
	reg op_carry, op_signed;
	reg op_add, op_sub;
	reg op_and, op_xor;
	reg op_eq, op_lt;
	reg op_ld, op_st;
	reg op_b, op_bc;

	reg [15:0] axi_addr;
	reg [7:0] axi_data;
	reg axi_aw, axi_w, axi_b, axi_ar, axi_r;

	assign mem_axi_awvalid = axi_aw;
	assign mem_axi_awaddr  = axi_addr;
	assign mem_axi_awprot  = 0;
	assign mem_axi_wvalid  = axi_w;
	assign mem_axi_wdata   = acc;
	assign mem_axi_bready  = axi_b;
	assign mem_axi_arvalid = axi_ar;
	assign mem_axi_araddr  = axi_addr;
	assign mem_axi_arprot  = 0;
	assign mem_axi_rready  = axi_r;

	task mem_read;
		input [15:0] addr;
		begin
			axi_addr <= addr;
			axi_ar <= 1;
		end
	endtask

	task mem_write;
		input [15:0] addr;
		begin
			axi_addr <= addr;
			axi_aw <= 1;
			axi_w <= 1;
		end
	endtask

	always @(posedge clk) begin
		if (!resetn) begin
			pc <= 256;
			{axi_aw, axi_w, axi_b, axi_ar, axi_r} <= 0;
			{op_add, op_sub, op_and, op_xor, op_eq, op_lt, op_ld, op_st, op_b, op_bc} <= 0;
			state <= state_exec;
		end else
		if (axi_ar || axi_r || axi_aw || axi_w || axi_b) begin
			if (axi_ar) begin
				if (mem_axi_arready) begin
					axi_ar <= 0;
					axi_r <= 1;
				end
			end
			if (axi_r) begin
				if (mem_axi_rvalid) begin
					axi_r <= 0;
					axi_data <= mem_axi_rdata;
				end
			end
			if (axi_aw || axi_w) begin
				if (mem_axi_awready) axi_aw <= 0;
				if (mem_axi_wready) axi_w <= 0;
				if ({axi_aw, axi_w} & ~{mem_axi_awready, mem_axi_wready} == 2'b00)
					axi_b <= 1;
			end
			if (axi_b) begin
				if (mem_axi_bvalid)
					axi_b <= 0;
			end
		end else
		case (state)
			state_decode: begin
				op_ind    <= axi_data[1:0];
				op_ind2   <= axi_data[1:0];
				op_carry  <= axi_data[2];
				op_signed <= axi_data[3];
				op_add    <= axi_data[7:3] == 'b00000;
				op_sub    <= axi_data[7:3] == 'b00001;
				op_eq     <= axi_data[7:3] == 'b00010;
				op_and    <= axi_data[7:2] == 'b000110;
				op_xor    <= axi_data[7:2] == 'b000111;
				op_lt     <= axi_data[7:4] == 'b0010;
				op_ld     <= axi_data[7:2] == 'b001100;
				op_st     <= axi_data[7:2] == 'b001101;
				op_b      <= axi_data[7:2] == 'b001110;
				op_bc     <= axi_data[7:2] == 'b001111;
				mem_read(pc);
				pc <= pc + 1;
				state <= state_arg1;
			end
			state_arg1: begin
				if (op_ind) begin
					mem_read(axi_data);
					op_ind <= op_ind-1;
				end else if (op_b || op_bc) begin
					if (op_b || !acc)
						pc[7:0] <= {8'bx, axi_data};
					else
						pc <= pc + 1;
					mem_read(pc);
					op_ind <= op_ind2;
					state <= state_arg2;
				end else if (op_ld) begin
					acc <= axi_data;
					mem_read(pc);
					pc <= pc + 1;
					state <= state_decode;
				end else if (op_st) begin
					mem_write(axi_data);
					state <= state_exec;
				end else
					state <= state_exec;
			end
			state_arg2: begin
				if (op_ind) begin
					mem_read(axi_data);
					op_ind <= op_ind-1;
				end else begin
					if (op_b || !acc) begin
						pc <= {axi_data, pc[7:0]} + 1;
						mem_read({axi_data, pc[7:0]});
					end else begin
						pc <= pc + 1;
						mem_read(pc);
					end
					state <= state_decode;
				end
			end
			state_exec: begin
				(* parallel_case *)
				case (1'b1)
					op_add: {carry, acc} <= acc + axi_data + (op_carry & carry);
					op_sub: {carry, acc} <= acc - axi_data + (op_carry & carry);
					op_and: acc <= acc & axi_data;
					op_xor: acc <= acc ^ axi_data;
					op_eq, op_lt: begin
						cmp_result = 0;

						(* parallel_case *)
						casez ({op_eq, op_lt, op_signed})
							3'b1zz: cmp_result = acc == axi_data;
							3'bz10: cmp_result = acc < axi_data;
							3'bz11: cmp_result = $signed(acc) < $signed(axi_data);
						endcase

						if (op_eq && op_carry) begin
							cmp_result = cmp_result && carry;
						end

						if (op_lt && op_carry) begin
							cmp_result = cmp_result || (carry && acc == axi_data);
						end

						acc <= cmp_result;
						carry <= cmp_result;
					end
				endcase

				mem_read(pc);
				pc <= pc + 1;
				state <= state_decode;
			end
		endcase
	end
endmodule
