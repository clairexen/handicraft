// http://www.clifford.at/yosys/vloghammer_bugs/issue_002_verilator.html
module test001;
wire [1:0] a = ~0;
wire [2:0] b = ~0;
wire [0:0] y;
assign y = $signed(a) == b;
initial if (y) $stop;
endmodule
