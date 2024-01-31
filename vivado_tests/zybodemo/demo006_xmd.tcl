
verbose

xload hw demo006.sdk/hw2/system.xml
source demo006.sdk/hw2/ps7_init.tcl

xconnect arm hw -cable type xilinx_tcf url TCP:127.0.0.1:3121
set_cur_target 64
set_cur_system
reset_zynqpl
xdisconnect 64
xdisconnect 352
set_cur_system_target

xfpga -cable type xilinx_tcf url TCP:127.0.0.1:3121 -f demo006.bit
xfpga_isconfigured -cable type xilinx_tcf url TCP:127.0.0.1:3121

xconnect arm hw -cable type xilinx_tcf url TCP:127.0.0.1:3121
set_cur_target 64
set_cur_system
xzynqresetstatus 64
ps7_init
ps7_post_config
xclearzynqresetstatus 64
xreset 64 0x80
xdownload 64 demo006.elf
xsafemode 64 off
xremove 64 all
xcontinue 64 0x100000 -status_on_stop

