module test(input [3:0] a, input [4:0] b, input [5:0] c, output [2:0] y);
assign y = {|a, |b, |c};
endmodule
