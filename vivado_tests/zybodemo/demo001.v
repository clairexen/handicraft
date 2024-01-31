`timescale 1ns / 1ps

module demo001_top(sw, led);
    input sw;
    output led;
    assign led = sw;
endmodule
