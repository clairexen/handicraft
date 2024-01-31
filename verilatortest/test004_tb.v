module test004_tb;
  reg a;
  wire [31:0] y;

  test004 UUT(a, y);

  initial begin
    a <= 0; #1;
    $display("%x", y);
    a <= 1; #1;
    $display("%x", y);
  end
endmodule
