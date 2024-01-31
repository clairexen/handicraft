module top(input [6:0] addr, output [15:0] romdata);
  romtab romtab_inst (.addr(addr), .romdata(romdata));
endmodule
