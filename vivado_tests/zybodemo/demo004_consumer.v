`timescale 1ns / 1ps

module demo004_consumer(aclk, aresetn, s_axis_tvalid, s_axis_tready, s_axis_tdata, state);
    
    // axis clock and reset
    input aclk, aresetn;
    
    // axis handshake
    input s_axis_tvalid;
    output s_axis_tready;
    
    // axis data
    input [7:0] s_axis_tdata;

    // the boolean value of the last input byte
    output reg state;
    
    always @(posedge aclk)
        if (s_axis_tvalid)
            state <= s_axis_tdata != 0;

    assign s_axis_tready = 1;

endmodule
