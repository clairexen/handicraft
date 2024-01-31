module demo2 (input clk, input [15:0] addr, output reg [7:0] dout);
  reg [7:0] romdata [0:65535];
  initial $readmemh("demo2.dat", romdata);
  always @(posedge clk) dout <= romdata[addr];
endmodule
