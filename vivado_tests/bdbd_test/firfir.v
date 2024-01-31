`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 11/12/2014 03:27:36 PM
// Design Name: 
// Module Name: firfir
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////
//Copyright 1986-2014 Xilinx, Inc. All Rights Reserved.
//--------------------------------------------------------------------------------
//Tool Version: Vivado v.2014.3 (lin64) Build 1034051 Fri Oct  3 16:31:15 MDT 2014
//Date        : Wed Nov 12 15:26:47 2014
//Host        : mwcomp2 running 64-bit Ubuntu 14.04.1 LTS
//Command     : generate_target design_1_wrapper.bd
//Design      : design_1_wrapper
//Purpose     : IP block netlist
//--------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

module firfir
   (M_AXIS_DATA_tdata,
    M_AXIS_DATA_tvalid,
    S_AXIS_DATA_tdata,
    S_AXIS_DATA_tready,
    S_AXIS_DATA_tvalid,
    aclk);
  output [31:0]M_AXIS_DATA_tdata;
  output M_AXIS_DATA_tvalid;
  input [15:0]S_AXIS_DATA_tdata;
  output S_AXIS_DATA_tready;
  input S_AXIS_DATA_tvalid;
  input aclk;

  wire [31:0]M_AXIS_DATA_tdata;
  wire M_AXIS_DATA_tvalid;
  wire [15:0]S_AXIS_DATA_tdata;
  wire S_AXIS_DATA_tready;
  wire S_AXIS_DATA_tvalid;
  wire aclk;

design_1 design_1_i
       (.M_AXIS_DATA_tdata(M_AXIS_DATA_tdata),
        .M_AXIS_DATA_tvalid(M_AXIS_DATA_tvalid),
        .S_AXIS_DATA_tdata(S_AXIS_DATA_tdata),
        .S_AXIS_DATA_tready(S_AXIS_DATA_tready),
        .S_AXIS_DATA_tvalid(S_AXIS_DATA_tvalid),
        .aclk(aclk));
endmodule
