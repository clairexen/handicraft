// yosys -p 'prep -top counter; async2sync; opt -nodffe -nosdff; show; write_smt2' counter.v
module counter(input clock, reset, up, output reg [7:0] count);
wire [7:0] t1 = count+1;
wire [7:0] t2 = up ? t1 : count;
wire [7:0] t3 = reset ? 8'h0 : t2;
always @(posedge clock) count <= t3;
endmodule
