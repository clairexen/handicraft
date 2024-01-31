
// little example for switch-level simulation in icarus verilog

module RES(input n1, output n2);
buf (weak1, weak0) g1 (n2, n1);
endmodule

module SW(input gate, inout cc1, inout cc2);
reg gate_buf;
always @(gate)
	if (gate === 0 || gate === 1)
		gate_buf = gate;
tranif1 sw (cc1, cc2, gate_buf);
endmodule

module testbench;

reg d, en;
wire q_tmp, q_bar, d_buf;
wire const0, const1;

assign const0 = 0, const1 = 1, d_buf = d;

SW sw_tr (en, d_buf, q_tmp);

SW sw_inv (q_tmp, const0, q_bar);
RES res_inv (const1, q_bar);

task step;
	input en_val, d_val;
	begin
		en <= en_val;
		d <= d_val;
		#10;
		$display("%b %b -> %b %b", en, d, q_tmp, q_bar);
	end
endtask

initial begin
	$dumpfile("example.vcd");
	$dumpvars(0, testbench);

	step(0, 0);
	step(1, 0);
	step(0, 0);

	$display("---");

	step(0, 1);
	step(1, 1);
	step(0, 1);

	$display("---");

	step(0, 0);
	step(1, 0);
	step(0, 0);
end

endmodule

