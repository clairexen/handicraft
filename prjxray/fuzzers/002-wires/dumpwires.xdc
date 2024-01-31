set_property -dict {PACKAGE_PIN F21 IOSTANDARD LVCMOS33} [get_ports A]
set_property -dict {PACKAGE_PIN G22 IOSTANDARD LVCMOS33} [get_ports Y]

set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]
set_property BITSTREAM.GENERAL.PERFRAMECRC YES [current_design]
