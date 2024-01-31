`timescale 1ns / 1ps

module minigpio (
    // Global Signals
    input aclk,
    input aresetn,

    // Write Address Channel
    input      [ 2:0] s_axi_awaddr,
    input             s_axi_awvalid,
    output reg        s_axi_awready,

    // Write Channel
    input      [31:0] s_axi_wdata,
    input             s_axi_wvalid,
    output reg        s_axi_wready,

    // Write Response Channel
    output reg [ 1:0] s_axi_bresp,
    output reg        s_axi_bvalid,
    input             s_axi_bready,

    // Read Address Channel
    input      [ 2:0] s_axi_araddr,
    input             s_axi_arvalid,
    output reg        s_axi_arready,

    // Read Channel
    output reg [31:0] s_axi_rdata,
    output reg [ 1:0] s_axi_rresp,
    output reg        s_axi_rvalid,
    input             s_axi_rready,

    // GPIO Pins
    input      [31:0] gpio_i,
    output reg [31:0] gpio_o
);

    // register file:
    //     0x00 ...... wr: gpio_o
    //                 rd: gpio_o
    //     0x04 ...... wr: gpio_o (xor)
    //                 rd: gpio_i

    always @(posedge aclk) begin
        s_axi_awready <= 0;
        s_axi_wready  <= 0;
        s_axi_arready <= 0;

        if (!aresetn) begin
            gpio_o <= 0;
            s_axi_bvalid <= 0;
            s_axi_rvalid <= 0;
        end else begin
            // handle axi4-lite writes
            if (s_axi_bready)
                s_axi_bvalid <= 0;
            if (s_axi_awvalid && s_axi_wvalid && !s_axi_bvalid) begin
                s_axi_awready <= 1;
                s_axi_wready  <= 1;
                s_axi_bresp   <= 0;
                s_axi_bvalid  <= 1;
                case (s_axi_awaddr)
                    0: gpio_o <= s_axi_wdata;
                    4: gpio_o <= gpio_o ^ s_axi_wdata;
                    default: s_axi_bresp <= 2;
                endcase
            end

            // handle axi4-lite reads
            if (s_axi_rready)
                s_axi_rvalid <= 0;
            if (s_axi_arvalid && !s_axi_rvalid) begin
                s_axi_arready <= 1;
                s_axi_rresp   <= 0;
                s_axi_rvalid  <= 1;
                case (s_axi_awaddr)
                    0: s_axi_rdata <= gpio_o;
                    4: s_axi_rdata <= gpio_i;
                    default: s_axi_bresp <= 2;
                endcase
            end
        end
    end

endmodule
