module test016_tb;

reg [1:0] a;
wire [8:0] y1;
wire [80:0] y2;

test016 uut (a, y1, y2);

initial begin
	a <= 00; #1; $display("%b %b", y1, y2);
	a <= 01; #1; $display("%b %b", y1, y2);
	a <= 10; #1; $display("%b %b", y1, y2);
	a <= 11; #1; $display("%b %b", y1, y2);
end

endmodule
