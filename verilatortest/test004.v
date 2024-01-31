module test004(a, y);
  input a;
  output [31:0] y;

  wire [7:0] y0;
  wire [7:0] y1;
  wire [7:0] y2;
  wire [7:0] y3;
  assign y = {y0,y1,y2,y3};

  localparam [7:0] v0 = +8'sd1 ** -8'sd2;
  localparam [7:0] v1 = +8'sd2 ** -8'sd2;
  localparam [7:0] v2 = -8'sd2 ** -8'sd3;
  localparam [7:0] v3 = -8'sd1 ** -8'sd3;
  localparam [7:0] zero = 0;

  assign y0 = a ? v0 : zero;
  assign y1 = a ? v1 : zero;
  assign y2 = a ? v2 : zero;
  assign y3 = a ? v3 : zero;
endmodule
