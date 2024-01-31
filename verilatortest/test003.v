// http://www.clifford.at/yosys/vloghammer_bugs/issue_005_verilator.html
module test003;
wire [3:0] a;
wire [3:0] y;
assign a = 4'b0010;
assign y = $signed(|a);
initial if (y != 4'b1111) $stop;
endmodule
