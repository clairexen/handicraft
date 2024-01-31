module demo02(input clk);
	reg [3:0] cnt_a = 1, cnt_b = 0;
	reg [7:0] counter = 150;

	always @(posedge clk) begin
		if (cnt_a) begin
			if (cnt_a == 10) begin
				cnt_a <= 0;
				cnt_b <= 1;
			end else
				cnt_a <= cnt_a + 1;
			counter <= counter + 1;
		end else begin
			if (cnt_b == 10) begin
				cnt_b <= 0;
				cnt_a <= 1;
			end else
				cnt_b <= cnt_b + 1;
			counter <= counter - 1;
		end
	end

	assert property (100 < counter && counter < 200);
	assert property (counter == cnt_a + 149 || counter == 161 - cnt_b);
	assert property ((cnt_a == 0) != (cnt_b == 0));
	assert property (cnt_a <= 10);
	assert property (cnt_b <= 10);

	always @* begin
		if (cnt_a)
			assert (counter == cnt_a + 149);
		else
			assert (counter == 161 - cnt_b);
	end
endmodule
