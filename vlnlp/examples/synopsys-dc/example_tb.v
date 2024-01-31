
`timescale 1ns / 1ps

module example_tb();

reg clk, rst, ctrl;
wire [3:0] out;

example UUT (
	.clk(clk),
	.rst(rst),
	.ctrl(ctrl),
	.step(3),
	.out(out)
);

initial begin
	clk <= 0;
	#100;
	forever begin
		clk <= !clk;
		#5;
	end
end

initial begin
	rst <= 1;
	#100;
	@(negedge clk);
	rst <= 0;
end

task count_show;
begin
	@(posedge clk);
	$display("  -> %d", out);
end
endtask

task count_up;
begin
	$display("=== UP ===");
	@(negedge clk);
	ctrl <= 1;
	@(negedge clk);
	@(negedge clk);
	@(negedge clk);
	ctrl <= 0;
	count_show;
end
endtask

task count_down;
begin
	$display("=== DOWN ===");
	@(negedge clk);
	ctrl <= 1;
	@(negedge clk);
	@(negedge clk);
	ctrl <= 0;
	@(negedge clk);
	count_show;
end
endtask

task count_reset;
begin
	$display("=== RESET ===");
	@(negedge clk);
	ctrl <= 1;
	@(negedge clk);
	ctrl <= 0;
	@(negedge clk);
	@(negedge clk);
	count_show;
end
endtask

initial begin
	#150;
	count_reset();
	count_up();
	count_up();
	count_down();
	count_up();
	count_reset();
	#100;
	$finish;
end

endmodule

