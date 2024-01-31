
module test002(a, b, y);

input [2:0] a;
input signed [1:0] b;
output y;

wire [2:0] b_buf = b;
assign y = a == ~b_buf;

endmodule

