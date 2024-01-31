
set_property -dict { IOSTANDARD LVCMOS33 PACKAGE_PIN W5  } [get_ports clk]
create_clock -add -name sys_clk_pin -period 10.00 -waveform {0 5} [get_ports clk]

set_property -dict { IOSTANDARD LVCMOS33 PACKAGE_PIN U16 } [get_ports {leds[0]}]
set_property -dict { IOSTANDARD LVCMOS33 PACKAGE_PIN E19 } [get_ports {leds[1]}]
set_property -dict { IOSTANDARD LVCMOS33 PACKAGE_PIN U19 } [get_ports {leds[2]}]
set_property -dict { IOSTANDARD LVCMOS33 PACKAGE_PIN V19 } [get_ports {leds[3]}]
set_property -dict { IOSTANDARD LVCMOS33 PACKAGE_PIN W18 } [get_ports {leds[4]}]
set_property -dict { IOSTANDARD LVCMOS33 PACKAGE_PIN U15 } [get_ports {leds[5]}]
set_property -dict { IOSTANDARD LVCMOS33 PACKAGE_PIN U14 } [get_ports {leds[6]}]
set_property -dict { IOSTANDARD LVCMOS33 PACKAGE_PIN V14 } [get_ports {leds[7]}]

set_property -dict { IOSTANDARD LVCMOS33 PACKAGE_PIN W7 } [get_ports {seg[0]}]
set_property -dict { IOSTANDARD LVCMOS33 PACKAGE_PIN W6 } [get_ports {seg[1]}]
set_property -dict { IOSTANDARD LVCMOS33 PACKAGE_PIN U8 } [get_ports {seg[2]}]
set_property -dict { IOSTANDARD LVCMOS33 PACKAGE_PIN V8 } [get_ports {seg[3]}]
set_property -dict { IOSTANDARD LVCMOS33 PACKAGE_PIN U5 } [get_ports {seg[4]}]
set_property -dict { IOSTANDARD LVCMOS33 PACKAGE_PIN V5 } [get_ports {seg[5]}]
set_property -dict { IOSTANDARD LVCMOS33 PACKAGE_PIN U7 } [get_ports {seg[6]}]
set_property -dict { IOSTANDARD LVCMOS33 PACKAGE_PIN V7 } [get_ports {seg[7]}]

set_property -dict { IOSTANDARD LVCMOS33 PACKAGE_PIN U2 } [get_ports {an[0]}]
set_property -dict { IOSTANDARD LVCMOS33 PACKAGE_PIN U4 } [get_ports {an[1]}]
set_property -dict { IOSTANDARD LVCMOS33 PACKAGE_PIN V4 } [get_ports {an[2]}]
set_property -dict { IOSTANDARD LVCMOS33 PACKAGE_PIN W4 } [get_ports {an[3]}]

set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]
set_property BITSTREAM.GENERAL.PERFRAMECRC YES [current_design]

