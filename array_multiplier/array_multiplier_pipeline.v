
module array_multiplier_pipeline(clk, a, b, y);

parameter width = 8;

input clk;
input [width-1:0] a, b;
output [width-1:0] y;

reg [width-1:0] a_pipeline [0:width-2];
reg [width-1:0] b_pipeline [0:width-2];
reg [width-1:0] partials [0:width-1];
integer i;

always @(posedge clk) begin
	a_pipeline[0] <= a;
	b_pipeline[0] <= b;
	for (i = 1; i < width-1; i = i+1) begin
		a_pipeline[i] <= a_pipeline[i-1];
		b_pipeline[i] <= b_pipeline[i-1];
	end

	partials[0] <= a[0] ? b : 0;
	for (i = 1; i < width; i = i+1)
		partials[i] <= (a_pipeline[i-1][i] ? b_pipeline[i-1] << i : 0) + partials[i-1];
end

assign y = partials[width-1];

endmodule

