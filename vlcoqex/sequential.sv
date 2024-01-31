module top (
	input clock, reset, up, down,
	output reg [7:0] count
);
	always @(posedge clock) begin
		if (reset)
			count <= 0;
		else if (up)
			count <= count + 1;
		else if (down)
			count <= count - 1;
	end
endmodule

module testbench (
	input clock, reset, up, down,
	output [7:0] count
);
	top uut (
		.clock(clock),
		.reset(reset),
		.up(up),
		.down(down),
		.count(count)
	);

	initial begin
		assume (reset);
	end

	always @(posedge clock) begin
		if (!reset) begin
			if (count == 0)
				assume (!down);
			if (count == 100)
				assume (!up);
			assert (count < 200);
		end
	end
endmodule
