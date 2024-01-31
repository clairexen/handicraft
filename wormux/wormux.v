module wormux1 #(
    parameter WIDTH = 32,
    parameter DEPTH = 10
) (
    input  [DEPTH - 1 : 0]           en,
    input  [(WIDTH * DEPTH) - 1 : 0] mux_in,
    output [WIDTH - 1 : 0]           mux_out
);
    reg [WIDTH - 1 : 0] bus;
    assign mux_out = bus;

    integer i;
    always @* begin
        bus = {WIDTH{1'b0}};
        for (i = 0; i < DEPTH; i = i + 1) begin
            if (en[i]) bus = bus | mux_in[WIDTH * i +: WIDTH];
        end
    end
endmodule

module wormux2 (
    input  [9:0]         en,
    input  [(10*32)-1:0] mux_in,
    output [31:0]        mux_out
);
    reg [31:0] bus;
    assign mux_out = bus;

    always @* begin
        bus = 32'h00000000;
        if (en[0]) bus = bus | mux_in[0*32 +: 32];
        if (en[1]) bus = bus | mux_in[1*32 +: 32];
        if (en[2]) bus = bus | mux_in[2*32 +: 32];
        if (en[3]) bus = bus | mux_in[3*32 +: 32];
        if (en[4]) bus = bus | mux_in[4*32 +: 32];
        if (en[5]) bus = bus | mux_in[5*32 +: 32];
        if (en[6]) bus = bus | mux_in[6*32 +: 32];
        if (en[7]) bus = bus | mux_in[7*32 +: 32];
        if (en[8]) bus = bus | mux_in[8*32 +: 32];
        if (en[9]) bus = bus | mux_in[9*32 +: 32];
    end
endmodule

module wormux3 (
    input  [9:0]         en,
    input  [(10*32)-1:0] mux_in,
    output [31:0]        mux_out
);
    reg [31:0] bus;
    assign mux_out = bus;

    always @* begin
        bus = 32'h00000000;
	(* parallel_case *)
        case (1'b1)
		en[0]: bus = mux_in[0*32 +: 32];
		en[1]: bus = mux_in[1*32 +: 32];
		en[2]: bus = mux_in[2*32 +: 32];
		en[3]: bus = mux_in[3*32 +: 32];
		en[4]: bus = mux_in[4*32 +: 32];
		en[5]: bus = mux_in[5*32 +: 32];
		en[6]: bus = mux_in[6*32 +: 32];
		en[7]: bus = mux_in[7*32 +: 32];
		en[8]: bus = mux_in[8*32 +: 32];
		en[9]: bus = mux_in[9*32 +: 32];
	endcase
    end
endmodule

module check_en (
	input  [9:0] en,
	output       ok
);
	assign ok = !en || (
		|(en & 10'b0101010101) != |(en & 10'b1010101010) &&
		|(en & 10'b0011001100) != |(en & 10'b1100110011) &&
		|(en & 10'b0000111100) != |(en & 10'b1111000011) &&
		|(en & 10'b0000000011) != |(en & 10'b1111111100)
	);
endmodule

module check_en_alt (
	input  [9:0] en,
	output       ok
);
	assign ok = !(en&(en-1));
endmodule

module check_en_alt2 (
	input  [9:0] en,
	output reg   ok
);
	integer i;
	reg [1:0] tmp;
	always @* begin
		tmp = 2'b11;
		for (i = 0; i < 10; i = i+1)
			tmp = tmp << en[i];
		ok = tmp[1];
	end
endmodule

module check_en_ref (
	input  [9:0] en,
	output reg   ok
);
	integer i, j;
	always @* begin
		j = 0;
		for (i = 0; i < 10; i = i+1)
			j = j + en[i];
		ok = j <= 1;
	end
endmodule

module miter12 (
	input  [9:0]         en,
	input  [(10*32)-1:0] mux_in,
	output               trigger
);
	wire [31:0] mux_out1, mux_out2;
	wire en_ok;

	check_en cen (en, en_ok);
	wormux1 uut1 (en, mux_in, mux_out1);
	wormux2 uut2 (en, mux_in, mux_out2);

	assign trigger = en_ok && mux_out1 != mux_out2;
endmodule

module miter13 (
	input  [9:0]         en,
	input  [(10*32)-1:0] mux_in,
	output               trigger
);
	wire [31:0] mux_out1, mux_out3;
	wire en_ok;

	check_en cen (en, en_ok);
	wormux1 uut1 (en, mux_in, mux_out1);
	wormux3 uut2 (en, mux_in, mux_out3);

	assign trigger = en_ok && mux_out1 != mux_out3;
endmodule
