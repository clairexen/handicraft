set_property -dict {PACKAGE_PIN L16 IOSTANDARD LVCMOS33} [get_ports clk]
set_property -dict {PACKAGE_PIN G15 IOSTANDARD LVCMOS33} [get_ports sw]
set_property -dict {PACKAGE_PIN M14 IOSTANDARD LVCMOS33} [get_ports led]
create_clock -period 8.00 [get_ports clk]
