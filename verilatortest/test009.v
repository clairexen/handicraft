module test009(output a, b, c, d, e, f, g);
 assign a = (5'b0 == (5'sb11111 >>> 3'd7));
 assign b = (5'sb11111 == (5'sb11111 >>> 3'd7));
 assign c = (1'b0+(5'sb11111 >>> 3'd7));
 assign d = (1'sb0+(5'sb11111 >>> 3'd7));
 assign e = (5'b0 == (5'sb11111 / 5'sd3));
 assign f = (5'sb0 == (5'sb11111 / 5'sd3));
 assign g = (5'b01010 == (5'b11111 / 5'sd3));
`ifndef VERILATOR
 initial #1 $display(a, b, c, d, e, f, g);
`endif
endmodule
