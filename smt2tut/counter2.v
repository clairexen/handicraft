module counter(input clk);
  reg  [2:0] cnt = 0;
  always @(posedge clk) cnt <= 2*cnt+3;
  assert property (cnt < 4);
endmodule
