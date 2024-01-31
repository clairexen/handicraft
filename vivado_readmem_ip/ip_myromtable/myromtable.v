module myromtable(input [6:0] addr, output [15:0] romdata);
  reg [15:0] rom [0:127];
  initial $readmemh("romdata.hex", rom);
  assign romdata = rom[addr];
endmodule
