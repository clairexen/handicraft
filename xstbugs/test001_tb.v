module test001_tb;

reg a;
wire [3:0] y;

test001 uut (.a(a), .y(y));

initial begin
	a <= 0; #1; $display("%b %b", a, y);
	a <= 1; #1; $display("%b %b", a, y);
	$finish;
end

endmodule
