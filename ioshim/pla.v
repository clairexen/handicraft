module pla #(
	parameter N_INPUTS = 20,
	parameter N_OUTPUTS = 20,
	parameter N_COLUMNS = 40
) (
	input [2*N_INPUTS*N_COLUMNS + N_COLUMNS*N_OUTPUTS - 1 : 0] cfg,
	input [N_INPUTS-1:0] din,
	output [N_OUTPUTS-1:0] dout
);
	wire [N_COLUMNS-1:0] cols;

	genvar i;
	generate for (i = 0; i < N_COLUMNS; i = i+1) begin:in2col
		wire [2*N_INPUTS-1:0] cfg_slice = cfg[2*N_INPUTS*i +: 2*N_INPUTS];
		assign cols[i] = &((cfg_slice[N_INPUTS-1:0] & din) | (cfg_slice[2*N_INPUTS-1:N_INPUTS] & ~din));
	end endgenerate
	generate for (i = 0; i < N_OUTPUTS; i = i+1) begin:col2out
		wire [N_COLUMNS-1:0] cfg_slice = cfg[2*N_INPUTS*N_COLUMNS + N_COLUMNS*i +: N_COLUMNS];
		assign dout[i] = |(cfg_slice & cols);
	end endgenerate
endmodule
