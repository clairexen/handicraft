module mux16to8 (input [7:0] secsa, minsa, secsb, minsb,
output reg [7:0] secs, mins,
input sela, selb, hold, seldisp,
output reg leda, ledb);
always @(*)
begin
    if (hold == 1'b1)
       if (seldisp == 1'b1) begin
        secs = secsa;
        mins = minsa;
        leda = 1'b0;
        ledb = 1'b1;
       end else begin
         secs = secsb;
         mins = minsb;
         leda = 1'b1;
         ledb = 1'b0;
    end
    else
        if (sela == 1'b1) begin
         secs = secsa;
         mins = minsa;
         leda = 1'b0;
          ledb = 1'b1;
        end else if (selb == 1'b1) begin
           secs = secsb;
           mins = minsb;
           leda = 1'b1;
           ledb = 1'b0;
          end else begin
           secs = 0;
          mins = 0;
         leda = 1'b1;
         ledb = 1'b1;
            end
     end
endmodule
