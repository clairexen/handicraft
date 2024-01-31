
`timescale 1 ns / 1 ps

module top(
  input         clk,
  input         s_axis_config_tvalid,
  output        s_axis_config_tready,
  input  [ 7:0] s_axis_config_tdata,
  input         s_axis_reload_tvalid,
  output        s_axis_reload_tready,
  input         s_axis_reload_tlast,
  input  [15:0] s_axis_reload_tdata,
  input         s_axis_data_tvalid,
  output        s_axis_data_tready,
  input  [23:0] s_axis_data_tdata,
  output        m_axis_data_tvalid,
  output [47:0] m_axis_data_tdata

);

fir_compiler_0 fir (
  .aclk(clk),
  .s_axis_config_tvalid(s_axis_config_tvalid),
  .s_axis_config_tready(s_axis_config_tready),
  .s_axis_config_tdata(s_axis_config_tdata),
  .s_axis_reload_tvalid(s_axis_reload_tvalid),
  .s_axis_reload_tready(s_axis_reload_tready),
  .s_axis_reload_tlast(s_axis_reload_tlast),
  .s_axis_reload_tdata(s_axis_reload_tdata),
  .s_axis_data_tvalid(s_axis_data_tvalid),
  .s_axis_data_tready(s_axis_data_tready),
  .s_axis_data_tdata(s_axis_data_tdata),
  .m_axis_data_tvalid(m_axis_data_tvalid),
  .m_axis_data_tdata(m_axis_data_tdata)
);

endmodule
