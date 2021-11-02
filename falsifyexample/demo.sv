module demo (
  input clk,
  output reg [5:0] counter
);
  initial counter = 0;

  always @(posedge clk) begin
    if (counter == 15)
`ifdef BUG1
      counter <= 16;
`else
      counter <= 0;
`endif
    else
`ifdef BUG2
      counter <= counter + 7;
`else
      counter <= counter + 1;
`endif
  end

`ifdef FORMAL
  always @(posedge clk) begin
    assert (counter < 32);
  end
`endif
endmodule
