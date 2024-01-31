set_property -dict { IOSTANDARD LVCMOS33 PACKAGE_PIN W5  } [get_ports clk]
set_property -dict { IOSTANDARD LVCMOS33 PACKAGE_PIN U16 } [get_ports led0]
set_property -dict { IOSTANDARD LVCMOS33 PACKAGE_PIN E19 } [get_ports led1]

create_clock -period 8.00 [get_ports clk]

set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]
