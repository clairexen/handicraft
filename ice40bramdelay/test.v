module test (input clk, output reg [7:0] leds);
	reg [7:0] rom [0:511];
	initial $readmemh("romdata.hex", rom);

	reg resetn = 0;
	reg [7:0] reset_count = 0;

	always @(posedge clk) begin
		if (reset_count == 35) // <-- set this to 36 to work around the issue
			resetn <= 1;
		reset_count <= reset_count + 1;
	end

	reg halt;
	reg [8:0] addr;

	always @(posedge clk) begin
		if (!resetn) begin
			halt <= 0;
			addr <= 0;
			leds <= ~0;
		end else if (!halt) begin
			halt <= !leds;
			addr <= addr + 1;
			leds <= rom[addr];
		end else begin
			leds <= 0;
		end
	end
endmodule
