`timescale 1ns / 1ps

module demo002_top(clk, sw, led);
    input clk, sw;
    output reg led = 0;

    // ZYBO System clock runs at 125 MHz: Toggle output every second
    parameter limit = 125000000;
    reg [$clog2(limit)-1:0] counter = 0;

    always @(posedge clk) begin
        if (counter < limit) begin
            if (sw)
                counter <= counter + 1;
        end else begin
            led <= !led;
            counter <= 0;
        end
    end
endmodule
