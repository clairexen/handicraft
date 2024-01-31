module top (
    input        sysclk,
    output [4:0] led,
    output [7:0] debug
);

parameter CLOCK_FREQ = 12000000;

wire count_tick;
freq_div #(.DIVISOR(CLOCK_FREQ/10)) f(
    .clk_in     ( sysclk ),
    .clk_out    ( count_tick )
);

reg [1:0] state = 0;
reg [3:0] led_shift = 1;
reg green_led = 0;
assign led = {green_led,led_shift};

parameter STATE_FORWARD   = 2'b01;
parameter STATE_BACKWARD  = 2'b10;

always @(posedge count_tick) begin
    state <= state;
    led_shift <= (state==STATE_FORWARD) ? (led_shift<<1) : (led_shift>>1);

    case(state)
    STATE_FORWARD: begin
        if ( led_shift == 4'b0100 ) begin
            state <= STATE_BACKWARD;
        end
    end
    STATE_BACKWARD: begin
        if ( led_shift == 4'b0010 ) begin
            state <= STATE_FORWARD;
        end
    end
    default: begin
        state <= STATE_FORWARD;
        led_shift <= 1;
        green_led = 1;
    end
    endcase
end

assign debug = { 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0};

endmodule
