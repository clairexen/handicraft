module test (
	output reg [7:0] o_downto,
	output reg [0:7] o_upto
);
	always @* begin
		o_downto = 0;
		o_upto = 0;

		// set the MSB bit (index 0 in edif array)
		o_downto[7] = 1;
		o_upto[0] = 1;
	end
endmodule
