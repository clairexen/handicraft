`timescale 1ns / 1ps

module demo003_top(clk, sw, led);
    input clk, sw;
    output reg led = 0;

    wire clk10mhz;
    wire pll_feedback;
    
    // Toggle every second
    parameter limit = 10000000;
    reg [$clog2(limit)-1:0] counter = 0;
    
    always @(posedge clk10mhz) begin
        if (counter < limit) begin
            if (sw)
                counter <= counter + 1;
        end else begin
            led <= !led;
            counter <= 0;
        end
    end
    
    // Based on the Vivado Language template for PLLE2_BASE
    // Note: the VCO freqency range is 800 MHz - 1866 MHz
    // See also: Xilinx UG472 (7 Series FPGAs Clocking Resources User Guide)
    //
    PLLE2_BASE #(
        .CLKFBOUT_MULT(10),       // Multiply value for all CLKOUT, (2-64)
        .CLKOUT0_DIVIDE(125),     // Divide amount for CLKOUT0 (1-128)
        .CLKOUT0_DUTY_CYCLE(0.5), // Duty cycle for CLKOUT0 (0.001-0.999).
        .CLKOUT0_PHASE(0.0),      // Phase offset for CLKOUT0 (-360.000-360.000).
        .STARTUP_WAIT("TRUE")     // Delay DONE until PLL Locks, ("TRUE"/"FALSE")
     )
     PLLE2_BASE_inst (
        // Input 125 MHz clock
        .CLKIN1(clk),
        
        // Generated 10 MHz clock 
        .CLKOUT0(clk10mhz),
        
        // Feedback
        .CLKFBOUT(pll_feedback),
        .CLKFBIN(pll_feedback),
        
        // Control Ports
        .PWRDWN(0),
        .RST(0)
     );
endmodule
