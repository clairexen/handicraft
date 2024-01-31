module shifter_reflect (input [31:0] H, L, input [4:0] shamt, input reverse, output [31:0] Y);
	wire [31:0] Hr, Lr, Yr;
	function [31:0] reflect;
		input [31:0] din;
		integer i;
		begin
			for (i = 0; i < 32; i=i+1)
				reflect[i] = din[31-i];
		end
	endfunction
	assign Hr = reverse ? reflect(H) : H;
	assign Lr = reverse ? reflect(L) : L;
	assign Yr = {Hr, Lr} >> shamt;
	assign Y = reverse ? reflect(Yr) : Yr;
endmodule
