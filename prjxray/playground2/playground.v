module top(input CLK, A, output Y);
  reg q1, q2;
  always @(posedge CLK) begin
  	q1 <= A;
  	q2 <= q1;
  end
  assign Y = q2;
endmodule
