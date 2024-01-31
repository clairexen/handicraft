module test016(a, y1, y2);
  input [1:0] a;
  output wire [8:0] y1;
  output reg [80:0] y2;

  assign y1 = { a == 2'b00, a == 2'b01, a == 2'b0x,
                a == 2'b10, a == 2'b11, a == 2'b1x,
                a == 2'bx0, a == 2'bx1, a == 2'bxx };

  function int_to_bit;
    input [1:0] i;
    begin
      int_to_bit = i[1] ? 1'bx : i[0];
    end
  endfunction

  integer i1, i2, i3, i4;
  initial begin
    for (i1 = 0; i1 < 3; i1 = i1+1)
    for (i2 = 0; i2 < 3; i2 = i2+1)
    for (i3 = 0; i3 < 3; i3 = i3+1)
    for (i4 = 0; i4 < 3; i4 = i4+1)
      y2[i1 + 3*i2 + 9*i3 + 27*i4] <= { int_to_bit(i1), int_to_bit(i2) } == { int_to_bit(i3), int_to_bit(i4) };
  end
endmodule
