// Arbitrary clock divider only even number dividors are supported.
//
//
// Usage example:
//  parameter sysclk_freq = 27'd12000000;  // in Hz for 12MHz clock
//  parameter CLK_DIVISOR = 64; // divide by 64 (optional, default is 4)
//  parameter CLK_DUTYCYCLE = 50; // duty cycle percent
//  parameter clk_freq    = sysclk_freq/CLK_DIVISOR;  // in Hz for 6MHz clock
//  wire clk;
//
//  // instantiate divider
//  freq_div #(.DIVISOR(CLK_DIVISOR),.DUTYCYCLE(CLK_DUTYCYCLE)) f(
//      .clk_in  ( sysclk ),
//      .clk_out ( clk )
//  );

module freq_div
#(
    parameter DIVISOR = 4, // default to division by 4
    parameter DUTYCYCLE = 50 // default to 50% duty cycle
)
(
    input      clk_in,
    output reg clk_out
);

reg [31:0] counter;
reg divided_clk;

initial begin
    divided_clk <= 1'b0;
    counter <= 32'b0;
end

always @(*) begin
    clk_out = (DIVISOR == 1 || DIVISOR == 0) ? clk_in: divided_clk;
end

always @ (posedge clk_in) begin
    if ((divided_clk == 1'b0) && (counter >= (DIVISOR*(100-DUTYCYCLE)/100-1))) begin // switch to duty high
        counter <= 32'b0;
        divided_clk <= 1'b1;
    end else if ((divided_clk == 1'b1) && (counter >= (DIVISOR*DUTYCYCLE/100-1))) begin // switch to duty low
        counter <= 32'b0;
        divided_clk <= 1'b0;
    end else begin // continue counting
        counter <= counter + 1;
        divided_clk <= divided_clk;
    end
end

endmodule
