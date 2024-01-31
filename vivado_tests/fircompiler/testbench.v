
`timescale 1 ns / 1 ps

module testbench;

reg clk;

reg s_axis_data_tvalid;
wire s_axis_data_tready;
reg [23:0] s_axis_data_tdata;

wire m_axis_data_tvalid;
wire signed [47:0] m_axis_data_tdata;

reg signed [63:0] outsample_ref, outsample_uut, outsample_diff;

fir_compiler_0 fir (
  .aclk(clk),
  .s_axis_config_tvalid(1'b0),
  .s_axis_reload_tvalid(1'b0),
  .s_axis_data_tvalid(s_axis_data_tvalid),
  .s_axis_data_tready(s_axis_data_tready),
  .s_axis_data_tdata(s_axis_data_tdata),
  .m_axis_data_tvalid(m_axis_data_tvalid),
  .m_axis_data_tdata(m_axis_data_tdata)
);

initial begin
  clk = 1;
  forever #2 clk = ~clk;
end

reg signed [63:0] num1, num2;
integer fd1, fd2, code1, code2;
integer cnt_clocks, cnt_samples, last_clocks;
reg [1023:0] buffer1, buffer2;

initial begin
  s_axis_data_tvalid <= 0;
  repeat(100) @(posedge clk);
  
  forever begin
    fd1 = $fopen(`SAMPLES_IN_TXT, "r");
    while (!$feof(fd1) && $fgets(buffer1, fd1))
    begin
      code1 = $sscanf(buffer1, "0x%x", num1);
      if (!code1) code1 = $sscanf(buffer1, "%d", num1);
      if (!code1) begin
        $display("Can't parse input sample value `%0s'.", buffer1);
        $finish;
      end
      
      s_axis_data_tvalid <= 1;
      s_axis_data_tdata <= num1;
      
      @(posedge clk);
      while (!s_axis_data_tready) @(posedge clk);
    end
    $fclose(fd1);
  end
end

initial begin
  repeat(100) @(posedge clk);
  
  forever begin
    fd2 = $fopen(`SAMPLES_OUT_TXT, "r");
    while (!$feof(fd2) && $fgets(buffer2, fd2))
    begin
      code2 = $sscanf(buffer2, "0x%x", num2);
      if (!code2) code2 = $sscanf(buffer2, "%d", num2);
      if (!code2) begin
        $display("Can't parse output sample value `%0s'.", buffer2);
        $finish;
      end
      
      @(posedge clk);
      while (!m_axis_data_tvalid) @(posedge clk);
      
      outsample_ref <= num2;
      outsample_uut <= m_axis_data_tdata;
      outsample_diff <= num2 - m_axis_data_tdata;
    end
    $fclose(fd2);
  end
end

initial begin
  last_clocks = 0;
  cnt_clocks = 0;
  cnt_samples = 0;
  forever begin
    @(posedge clk);
    cnt_clocks = cnt_clocks + 1;
    if (m_axis_data_tvalid) begin
      $display("Got output sample #%0d = %x after %0d clocks (delta=%0d)", cnt_samples, m_axis_data_tdata, cnt_clocks, cnt_clocks - last_clocks);
      cnt_samples = cnt_samples + 1;
      last_clocks = cnt_clocks;
    end
  end 
end


endmodule
