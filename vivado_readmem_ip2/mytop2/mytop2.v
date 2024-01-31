module mytop2(input [6:0] addr, output [15:0] romdata);
  myip2_0 myip2_0_inst (.addr(addr), .romdata(romdata));
endmodule
