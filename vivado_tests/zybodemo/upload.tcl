# Usage example:
# /opt/Xilinx/Vivado/2014.1/bin/vivado -mode batch -source upload.tcl -tclargs demo001.bit

connect_hw_server
open_hw_target [lindex [get_hw_targets] 0]
set_property PROGRAM.FILE $argv [lindex [get_hw_devices] 1]
program_hw_devices [lindex [get_hw_devices] 1]
