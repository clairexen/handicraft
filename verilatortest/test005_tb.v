module test005_tb;
  reg [3:0] a;
  wire [3:0] y;
  integer i;

  test005 UUT(a, y);

  initial begin
    for (i=0; i < 16; i=i+1) begin
      a <= i; #1;
      $display("%x %x", a, y);
    end
  end
endmodule
