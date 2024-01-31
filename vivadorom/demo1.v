module demo1 (input clk, input [15:0] addr, output reg [7:0] dout);
  reg [7:0] romdata [0:65535];
  initial $readmemh("demo1.dat", romdata);
  always @(posedge clk) dout <= romdata[addr];
endmodule
