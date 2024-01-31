module mem_with_maybe_reset (
        input clk, resetn, wen,
        input [4:0] waddr, raddr,
        input [31:0] wdata,
        output reg [31:0] rdata
);
        parameter ZERO_RESET = 0;

        integer i;
        reg [31:0] mem [0:31];

        always @(posedge clk) begin
                if (!resetn) begin
                        if (ZERO_RESET)
                             for (i = 0; i < 32; i = i+1)
                                     mem[i] <= 0;
                end else begin
                        if (wen)
                                mem[waddr] <= wdata;
                        rdata <= mem[raddr];
                end
        end
endmodule
