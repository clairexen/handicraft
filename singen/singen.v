module singen(input clk, output reg out_sq, out_pwm, out_dm);

	// Sine Table

	reg [15:0] sintab [0:1023];
	initial $readmemh("sintable.hex", sintab);

	reg [13:0] counter = 0;
	reg [15:0] sample;

	always @(posedge clk) begin
		counter <= counter + 1;
		sample <= sintab[counter >> 4];
	end

	// Square Wave Output

	always @(posedge clk) begin
		out_sq <= sample > 16'h8000;
	end

	// PWM Output

	always @(posedge clk) begin
		out_pwm <= (sample >> 12) > counter[3:0];
	end

	// Delta Modulation Output

	reg [15:0] dm_state = 0;
	localparam [15:0] dm_delta = 12;

	always @(posedge clk) begin
		if (out_dm) begin
			if (dm_state <= (16'hffff - dm_delta))
				dm_state <= dm_state + dm_delta;
		end else begin
			if (dm_state >= dm_delta)
				dm_state <= dm_state - dm_delta;
		end
		out_dm <= sample > dm_state;
	end

endmodule
