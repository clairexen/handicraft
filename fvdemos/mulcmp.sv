module mulcmp #(
	parameter integer IBITS = 16,
	parameter integer PBITS = 31
) (
	input [IBITS-1:0] a, b, c, d
);
	wire [PBITS-1:0] product_ab = a * b;
	wire [PBITS-1:0] product_cd = c * d;

	always_comb begin
		assume (a < c);
		assume (b < d);
		assert (product_ab < product_cd);
	end
endmodule
