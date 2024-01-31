module test007_tb;
  reg [4:0] a;
  reg [2:0] b;
  wire [3:0] y;
  integer i;
  test008 uut (a, b, y);
  initial
    for (i = 0; i < 256; i = i+1) begin
      {a, b} <= i; #1; $write("%x", y);
      if (i % 16 == 15) $display;
    end
endmodule
