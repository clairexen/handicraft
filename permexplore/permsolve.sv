module permcluster #(
	parameter N = 6, // number of operations
	parameter M = 5  // number of constraints
);
	rand const reg [2*N-1:0] ops;
	rand const reg [5*N-1:0] args;

	reg [5*M-1:0] dins;
	wire [5*M-1:0] douts;

	permchain #(.N(N)) chains [0:M-1] (
		.ops(ops),
		.args(args),
		.din(dins),
		.dout(douts)
	);

	task constr;
		input integer k, i, j;
		begin
			dins[5*k +: 5] = j;
			assume(douts[5*k +: 5] == i);
		end
	endtask

	always @* begin
		// RISC-V CJ-type immediate
		//constr(10, 10, 12);
		//constr( 9,  9,  8);
		//constr( 8,  8, 10);
		//constr( 7,  7,  9);
		//constr( 6,  6,  6);
		//constr( 5,  5,  7);
		constr( 4,  4,  2);
		constr( 3,  3, 11);
		constr( 2,  2,  5);
		constr( 1,  1,  4);
		constr( 0,  0,  3);
		cover(1);
	end

	// some obvious symmetry breaking
	integer i;
	always @* begin
		for (i = 0; i < N-1; i = i+1) begin
			// GREV is only allowed after ROT (or as first op)
			if (ops[2*i +: 2] != 0) assume(ops[2*(i+1) +: 2] != 1);

			// ROT is not allowed after ROT
			if (ops[2*i +: 2] == 0) assume(ops[2*(i+1) +: 2] != 0);
		end
	end
endmodule

module permchain #(
	parameter N = 2  // number of operations
) (
	input [2*N-1:0] ops,
	input [5*N-1:0] args,
	input [4:0] din,
	output [4:0] dout
);
	wire [5*N-1:0] dins, douts;

	assign dins[4:0] = din;
	assign dins[5*N-1:5] = douts[5*N-6:0];
	assign dout = douts[5*N-1:5*N-5];

	permop ops [0:N-1] (
		.op(ops),
		.arg(args),
		.din(dins),
		.dout(douts)
	);
endmodule

module permop (
	input [1:0] op,
	input [4:0] arg,
	input [4:0] din,
	output [4:0] dout
);
	function [4:0] revert;
		input [4:0] d;
		begin
			revert[0] = d[4];
			revert[1] = d[3];
			revert[2] = d[2];
			revert[3] = d[1];
			revert[4] = d[0];
		end
	endfunction

	function [4:0] unperm;
		input [3:0] k;
		input [4:0] i;
		begin
			unperm = ((k + (i & k & ~(k<<1))) & ~k) | (i & ~(k | (k<<1))) | ((i>>1) & k);
		end
	endfunction

	wire [4:0] dout_ror = din-arg;
	wire [4:0] dout_grev = din^arg;
	wire [4:0] dout_shfl = revert(unperm(revert(arg), revert(din)));
	wire [4:0] dout_unshfl = unperm(arg, din);

	assign dout = op == 0 ? dout_ror : op == 1 ? dout_grev :
			op == 2 ? dout_shfl : dout_unshfl;
endmodule
