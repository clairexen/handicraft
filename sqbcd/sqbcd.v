// solving the sat problem (yosys only):
//   yosys -tl yosys.log -p "synth -top top; flatten; opt; abc;; sat -set y 0 -set ok 1 -show a,b top" sqbcd.v
//
// solving the sat problem (vivado + yosys):
//   Vivado% read_verilog ./sqbcd.v 
//   Vivado% synth_design -part xc7k70t -top top
//   Vivado% write_verilog synth.v
//   yosys -tl yosys.log -p "hierarchy -top top; techmap; opt top %n; flatten; opt; abc;; sat -set y 0 -set ok 1 -show a,b top" synth.v xlcells.v
//
// finding the wanted number:
//   - get the digits of 'a': rpn -c 0b<bits_of_a>
//   - get the digits of 'b': rpn -c 0b<bits_of_b>
//   - take the sqaure of "aaaa.0123456789bbbb"

module square_bcd (a, y);
  parameter digits = 4;
  input [4*digits-1:0] a;
  output reg [4*digits-1:0] y;
  reg [6:0] t;

  integer i, j;
  always @* begin
    y = 0;
    for (i = 0; i < digits; i=i+1) begin
      t = 0;
      for (j = 0; j < digits-i; j=j+1) begin
        t = t + a[4*i +: 4] * a[4*j +: 4] + y[4*(i+j) +: 4];
        y[4*(i+j) +: 4] = t % 7'd10;
	t = t / 7'd10;
      end
    end
  end
endmodule

module top(a, b, y, ok);
  localparam int_digits = 15, frac_digits = 10;
  localparam [4*frac_digits-1:0] forced_frac = 'h0123456789;

  // (aaaa.0123456789bbbb)^2 = <some_int> + 0.yyyyyyyy
  input [4*int_digits-1:0] a;
  input [4*frac_digits-1:0] b;
  output [8*frac_digits-1:0] y;
  output reg ok;

  wire [16*frac_digits-1:0] x;
  wire [4*int_digits+8*frac_digits-1:0] n = { a, forced_frac, b };
  square_bcd #(4*frac_digits) sq (n, x);
  assign y = x[8*frac_digits +: 8*frac_digits];

  integer i;
  always @* begin
    ok = 1;
    for (i = 0; i < int_digits; i=i+1)
      if (a[4*i +: 4] >= 10) ok = 0;
    for (i = 0; i < frac_digits; i=i+1)
      if (b[4*i +: 4] >= 10) ok = 0;
  end
endmodule
