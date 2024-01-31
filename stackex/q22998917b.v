module mux16to8 (input [7:0] secsa, minsa, secsb, minsb,
output [7:0] secs, mins,
input sela, selb, hold, seldisp,
output leda, ledb);

assign {secs, mins, leda, ledb} =
    hold ? seldisp ? {secsa, minsa, 1'b0, 1'b1} :
                     {secsb, minsb, 1'b1, 1'b0} :
              sela ? {secsa, minsa, 1'b0, 1'b1} :
              selb ? {secsb, minsb, 1'b1, 1'b0} :
                     { 8'd0,  8'd0, 1'b1, 1'b1};

endmodule
