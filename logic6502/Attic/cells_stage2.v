
// -------------------------------------------------------

module pull_up(output y);
wire vcc = 1;
RES res (vcc, y);
endmodule

// -------------------------------------------------------

module switch_up(input gate, output y);
wire vcc = 1;
SW sw (gate, vcc, y);
endmodule

// -------------------------------------------------------

module switch_down(input gate, output y);
wire vss = 0;
SW sw (gate, vss, y);
endmodule

// -------------------------------------------------------

