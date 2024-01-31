module testmodule(a, y);

input [7:0] a;
output y;
assign y = ^a;

endmodule
