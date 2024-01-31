module mytop(input [6:0] addr, output [15:0] romdata);
  myip1_0 myip1_0_inst (.addr(addr), .romdata(romdata));
endmodule
