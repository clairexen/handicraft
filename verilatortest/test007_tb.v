module test007_tb;
  reg [3:0] a;
  wire [4:0] y;
  integer i;
  test007 uut (a, y);
  initial
    for (i = 0; i < 16; i = i+1) begin
      a <= i; #1; $display("%x -> %x", a, y);
    end
endmodule
