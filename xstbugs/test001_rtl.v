
// Bug in XST 14.5, found by Clifford Wolf using yosys and xsthammer.
// ------------------------------------------------------------------
//
// xst sets 'y' to the 4-bit sign-extended value of 'a'. but instead
// it should zero extend 'a'. See Sec. 5.5.1 of IEEE Std. 1364-255:
//
//   "Concatenate results are unsigned, regardless of the operands."
//
// (Sec. 5.1.14 states that replications are concatenations.)
//
// XST truth-table (incorrect behavior):
//   0 0000
//   1 1111
//
// ISIM truth-table (correct behavior):
//   0 0000
//   1 0001

module test001(a, y);
input signed [0:0] a;
output [3:0] y;
assign y = {1{a}};
endmodule

