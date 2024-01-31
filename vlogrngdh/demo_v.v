module test;
	function [31:0] myrand;
		input [31:0] v1;
		input [31:0] v2;
		begin
			myrand = v1;
			myrand = myrand ^ (myrand << 13);
			myrand = myrand ^ (myrand >> 17);
			myrand = myrand ^ (myrand << 5);
			myrand = myrand ^ v2;
			myrand = myrand ^ (myrand << 13);
			myrand = myrand ^ (myrand >> 17);
			myrand = myrand ^ (myrand << 5);
		end
	endfunction
	initial begin
		$display("%x", myrand($random, $random));
		$display("%x", myrand($random, $random));
		$display("%x", myrand($random, $random));
		$finish;
	end
endmodule
