`timescale 1ns / 1ps

module demo004_generator(aclk, aresetn, m_axis_tvalid, m_axis_tready, m_axis_tdata);

    parameter [31:0] limit = 100000000;
    
    // axis clock and reset
    input aclk, aresetn;
    
    // axis handshake
    output reg  m_axis_tvalid;
    input m_axis_tready;
    
    // axis data
    output reg [7:0] m_axis_tdata;
    
    reg [31:0] counter;
    
    always @(posedge aclk) begin
    
        // synchronous reset
        if (!aresetn) begin
            m_axis_tvalid <= 0;
            m_axis_tdata <= 0;
            counter <= 0;
        end else begin
            // not transmitting? wait for counter
            if (!m_axis_tvalid) begin
                counter <= counter + 1;
                if (counter == limit)
                    m_axis_tvalid <= 1;
            end
    
            // transmitting? wait for slave to read
            if (m_axis_tvalid && m_axis_tready) begin
                m_axis_tdata <= !m_axis_tdata;
                m_axis_tvalid <= 0;
                counter <= 0;
            end
        end
    end

endmodule
