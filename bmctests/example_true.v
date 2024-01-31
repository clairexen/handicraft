module main(input clk, input [2:0] select);
  reg [4:0] addr = 0;
  reg [7:0] counter = 1;
  reg [7:0] memory [0:31];
  reg [7:0] t1, t2;

  always @(posedge clk) begin
    addr <= addr + 1;
    memory[addr] <= counter;
    
    case (select[2])
      0: t1 = addr > 0 ? memory[addr >> 1] : 13;
      1: t1 = addr > 0 ? memory[addr -  1] : 17;
    endcase

    case (select[1:0])
      0: t2 = 3;
      1: t2 = 5;
      2: t2 = 7;
      3: t2 = 11;
    endcase

    counter <= counter + t1 + t2;
  end

  assert property (counter[0]);
endmodule
