set_property -dict {PACKAGE_PIN F21 IOSTANDARD LVCMOS33} [get_ports {PI[7]}]
set_property -dict {PACKAGE_PIN G22 IOSTANDARD LVCMOS33} [get_ports {PI[6]}]
set_property -dict {PACKAGE_PIN G21 IOSTANDARD LVCMOS33} [get_ports {PI[5]}]
set_property -dict {PACKAGE_PIN D21 IOSTANDARD LVCMOS33} [get_ports {PI[4]}]
set_property -dict {PACKAGE_PIN E21 IOSTANDARD LVCMOS33} [get_ports {PI[3]}]
set_property -dict {PACKAGE_PIN D22 IOSTANDARD LVCMOS33} [get_ports {PI[2]}]
set_property -dict {PACKAGE_PIN E22 IOSTANDARD LVCMOS33} [get_ports {PI[1]}]
set_property -dict {PACKAGE_PIN A21 IOSTANDARD LVCMOS33} [get_ports {PI[0]}]
set_property -dict {PACKAGE_PIN B21 IOSTANDARD LVCMOS33} [get_ports {PO[7]}]
set_property -dict {PACKAGE_PIN B22 IOSTANDARD LVCMOS33} [get_ports {PO[6]}]
set_property -dict {PACKAGE_PIN C22 IOSTANDARD LVCMOS33} [get_ports {PO[5]}]
set_property -dict {PACKAGE_PIN C20 IOSTANDARD LVCMOS33} [get_ports {PO[4]}]
set_property -dict {PACKAGE_PIN D20 IOSTANDARD LVCMOS33} [get_ports {PO[3]}]
set_property -dict {PACKAGE_PIN F20 IOSTANDARD LVCMOS33} [get_ports {PO[2]}]
set_property -dict {PACKAGE_PIN F19 IOSTANDARD LVCMOS33} [get_ports {PO[1]}]
set_property -dict {PACKAGE_PIN A19 IOSTANDARD LVCMOS33} [get_ports {PO[0]}]
set_property -dict {PACKAGE_PIN J20 IOSTANDARD LVCMOS33} [get_ports CLK]

set_property LOCK_PINS {I0:A1 I1:A2 I2:A3 I3:A4 I4:A5 I5:A6} [get_cells -hierarchical -filter {REF_NAME == LUT6}]

set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]
set_property BITSTREAM.GENERAL.PERFRAMECRC YES [current_design]

# set_property -dict {IS_LOC_FIXED 1 IS_BEL_FIXED 1 LOC SLICE_X16Y130 BEL SLICEL.A6LUT} [get_cells LUT_A]
# set_property -dict {IS_LOC_FIXED 1 IS_BEL_FIXED 1 LOC SLICE_X16Y130 BEL SLICEL.B6LUT} [get_cells LUT_B]
# set_property -dict {IS_LOC_FIXED 1 IS_BEL_FIXED 1 LOC SLICE_X16Y130 BEL SLICEL.C6LUT} [get_cells LUT_C]
# set_property -dict {IS_LOC_FIXED 1 IS_BEL_FIXED 1 LOC SLICE_X16Y130 BEL SLICEL.D6LUT} [get_cells LUT_D]
#
# set_property FIXED_ROUTE {CLBLL_LL_A CLBLL_LL_AMUX CLBLL_LOGIC_OUTS20 SS6BEG2 SW6BEG2} [get_nets OA_OBUF]

create_pblock pblock
add_cells_to_pblock [get_pblocks pblock] [get_cells playground]
resize_pblock [get_pblocks pblock] -add {SLICE_X20Y135:SLICE_X23Y136}

set_property CONTAIN_ROUTING 1 [get_pblocks pblock]
set_property EXCLUDE_PLACEMENT 1 [get_pblocks pblock]

