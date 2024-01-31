
// -------------------------------------------------------

module dlatch_not(en, a, y, q);

input en, a;
output y, q;

(* nmos_sibblings = 2 *)
PULLUP p1 (.y(y));

(* nmos_sibblings = 2 *)
SW0 s0 (.gate(a), .cc(y));

(* nmos_sibblings = 2 *)
SW s1 (.gate(en), .cc1(y), .cc2(q));

endmodule

// -------------------------------------------------------

module dlatch_nor2(en, a, b, y, q);

input en, a, b;
output y, q;

(* nmos_sibblings = 3 *)
PULLUP p1 (.y(y));

(* nmos_sibblings = 3 *)
SW0 s0 (.gate(a), .cc(y)),
    s1 (.gate(b), .cc(y));

(* nmos_sibblings = 3 *)
SW s2 (.gate(en), .cc1(y), .cc2(q));

endmodule

// -------------------------------------------------------

module dlatch_nor3(en, a, b, c, y, q);

input en, a, b, c;
output y, q;

(* nmos_sibblings = 4 *)
PULLUP p1 (.y(y));

(* nmos_sibblings = 4 *)
SW0 s0 (.gate(a), .cc(y)),
    s1 (.gate(b), .cc(y)),
    s2 (.gate(c), .cc(y));

(* nmos_sibblings = 4 *)
SW s3 (.gate(en), .cc1(y), .cc2(q));

endmodule

// -------------------------------------------------------

module dlatch_nor4(en, a, b, c, d, y, q);

input en, a, b, c, d;
output y, q;

(* nmos_sibblings = 5 *)
PULLUP p1 (.y(y));

(* nmos_sibblings = 5 *)
SW0 s0 (.gate(a), .cc(y)),
    s1 (.gate(b), .cc(y)),
    s2 (.gate(c), .cc(y)),
    s3 (.gate(d), .cc(y));

(* nmos_sibblings = 5 *)
SW s4 (.gate(en), .cc1(y), .cc2(q));

endmodule

// -------------------------------------------------------

module buffer(en0, en1, q);

input en0, en1;
output q;

(* nmos_sibblings = 1 *)
SW0 s0 (.gate(en0), .cc(q));

(* nmos_sibblings = 1 *)
SW1 s1 (.gate(en1), .cc(q));

endmodule

// -------------------------------------------------------

