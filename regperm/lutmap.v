module \$lut (A, Y);
  parameter WIDTH = 0;
  parameter LUT = 0;

  (* force_downto *)
  input [WIDTH-1:0] A;
  output Y;

  generate
    if (WIDTH == 1) begin
      LUT1 #(.INIT(LUT)) _TECHMAP_REPLACE_ (.Y(Y), .A(A[0]));
    end else
    if (WIDTH == 2) begin
      LUT2 #(.INIT(LUT)) _TECHMAP_REPLACE_ (.Y(Y), .A(A[0]), .B(A[1]));
    end else
    if (WIDTH == 3) begin
      LUT3 #(.INIT(LUT)) _TECHMAP_REPLACE_ (.Y(Y), .A(A[0]), .B(A[1]), .C(A[2]));
    end else
    if (WIDTH == 4) begin
      LUT4 #(.INIT(LUT)) _TECHMAP_REPLACE_ (.Y(Y), .A(A[0]), .B(A[1]), .C(A[2]), .D(A[3]));
    end else begin
      wire _TECHMAP_FAIL_ = 1;
    end
  endgenerate
endmodule
