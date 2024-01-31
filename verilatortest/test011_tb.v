module test011_tb;
  reg [3:0] a;
  wire [44:0] y;
  test011 uut (a, a, a, a, y);
  initial begin
    a <=  0; #1; $display("%b", y[37:34]);
    a <= 15; #1; $display("%b", y[37:34]);
  end
endmodule
