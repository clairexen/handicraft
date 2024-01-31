
module test002_tb;

reg [2:0] a;
reg [1:0] b;
wire y;

test002 uut (.a(a), .b(b), .y(y));

integer i;
initial begin
	for (i = 0; i < 32; i = i+1) begin
		{a, b} <= i; #1;
		if (y)
			$display("%b %b %b", a, b, y);
	end
	$finish;
end

endmodule

