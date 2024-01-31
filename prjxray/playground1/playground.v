module top (input CLK, input [7:0] PI, output reg [7:0] PO);
	reg [7:0] PI_Q;
	wire [7:0] PO_D;

	always @(posedge CLK) begin
		PI_Q <= PI;
		PO <= PO_D;
	end

	playground playground (CLK, PI_Q, PO_D);
endmodule

(* DONT_TOUCH = "yes" *)
module playground (input CLK, input [7:0] PI, output [7:0] PO);
	(* DONT_TOUCH = "yes" *) LUT6 #(
		.INIT(64'h 0123_4567_89ab_cdef)
	) LUTS[7:0] (
		.I0({PI[0], PI[1], PI[2], PI[3], PI[4], PI[5], PI[6], PI[7]}),
		.I1({PI[1], PI[2], PI[3], PI[4], PI[5], PI[6], PI[7], PI[0]}),
		.I2({PI[2], PI[3], PI[4], PI[5], PI[6], PI[7], PI[0], PI[1]}),
		.I3({PI[3], PI[4], PI[5], PI[6], PI[7], PI[0], PI[1], PI[2]}),
		.I4({PI[4], PI[5], PI[6], PI[7], PI[0], PI[1], PI[2], PI[3]}),
		.I5({PI[5], PI[6], PI[7], PI[0], PI[1], PI[2], PI[3], PI[4]}),
		.O(PO[7:0])
	);
endmodule
