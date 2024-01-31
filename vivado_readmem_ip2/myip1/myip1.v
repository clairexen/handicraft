module myip1(input [6:0] addr, output [15:0] romdata);
  reg [15:0] rom [0:127];
  initial $readmemh("myip1.dat", rom);
  assign romdata = rom[addr];
endmodule
