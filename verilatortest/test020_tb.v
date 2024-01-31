module test020_tb;
  reg [2:0] a;
  wire [3:0] y;

  test020 uut (a, y);

  integer i;
  initial for (i = 0; i < 8; i = i+1) begin
      a = i; #1; $display("%d %b", a, y);
    end
endmodule
