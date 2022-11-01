module testbench (input clk);
	integer seq = 0;
	always @(posedge clk) seq <= seq+1;

	(* keep *) wire cnt;
	reg rst, up3, dn2;
	always @* begin
		case (seq)
			0: {rst, up3, dn2} <= 3'b 100;
			1: {rst, up3, dn2} <= 3'b 010;
			2: {rst, up3, dn2} <= 3'b 001;
			3: {rst, up3, dn2} <= 3'b 011;
			4: {rst, up3, dn2} <= 3'b 000;
			default: {rst, up3, dn2} <= 3'b 100;
		endcase
	end

	top uut (
		.clk(clk),
		.rst(rst),
		.up3(up3),
		.dn2(dn2),
		.cnt(cnt)
	);
endmodule
