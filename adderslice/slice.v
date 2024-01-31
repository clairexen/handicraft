module alu_slice(input A, B, CI, M, output S, CO);
  assign {CO,S} = A + (B^M) + CI;
endmodule
