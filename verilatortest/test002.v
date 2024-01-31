// http://www.clifford.at/yosys/vloghammer_bugs/issue_016_verilator.html
module test002;
wire [3:0] y0;
wire [3:0] y1;
assign y0  = -4'd1 ** -4'sd2;
assign y1  = -4'd1 ** -4'sd3;
initial if (y0 || y1) $stop;
endmodule
