#!/bin/bash
#
# Installing iCEcube2:
#  - Install iCEcube2.2014.08 in /opt/lscc/iCEcube2.2014.08
#  - Install License in /opt/lscc/iCEcube2.2014.08/license.dat
#
# Creating a project:
#  - <project_name>.v    ## HDL sources (use "top" as name for the top module)
#  - <project_name>.sdc  ## timing constraint file
#  - <project_name>.pcf  ## physical constraint file
#
# Running iCEcube2:
#  - bash icecuberun.sh <project_name>  ## creates <project_name>.bin
#

set -ex
icecubedir="/opt/lscc/iCEcube2.2014.08"
export SBT_DIR="$icecubedir/sbt_backend"
export SYNPLIFY_PATH="$icecubedir/synpbase"
export LM_LICENSE_FILE="$icecubedir/license.dat"
export TCL_LIBRARY="$icecubedir/sbt_backend/bin/linux/lib/tcl8.4"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH${LD_LIBRARY_PATH:+:}$icecubedir/sbt_backend/bin/linux/opt"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH${LD_LIBRARY_PATH:+:}$icecubedir/sbt_backend/bin/linux/opt/synpwrap"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH${LD_LIBRARY_PATH:+:}$icecubedir/sbt_backend/lib/linux/opt"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH${LD_LIBRARY_PATH:+:}$icecubedir/LSE/bin/lin"

(
rm -rf "$1.tmp"
mkdir -p "$1.tmp"
cp "$1.v" "$1.tmp/input.v"
if test -f "$1.sdc"; then cp "$1.sdc" "$1.tmp/input.sdc"; fi
if test -f "$1.pcf"; then cp "$1.pcf" "$1.tmp/input.pcf"; fi
cd "$1.tmp"

touch input.sdc
touch input.pcf

mkdir -p outputs/bitmap
mkdir -p outputs/netlister
mkdir -p outputs/packer
mkdir -p outputs/placer
mkdir -p outputs/router
mkdir -p outputs/simulation_netlist
mkdir -p outputs/timer

cat > impl_syn.prj << EOT
add_file -verilog -lib work input.v
impl -add impl -type fpga

# implementation attributes
set_option -vlog_std v2001
set_option -project_relative_includes 1

# device options
set_option -technology SBTiCE40
set_option -part iCE40HX1K
set_option -package TQ144
set_option -speed_grade
set_option -part_companion ""

# mapper_options
set_option -frequency auto
set_option -write_verilog 0
set_option -write_vhdl 0

# Silicon Blue iCE40
set_option -maxfan 10000
set_option -disable_io_insertion 0
set_option -pipe 1
set_option -retiming 0
set_option -update_models_cp 0
set_option -fixgatedclocks 2
set_option -fixgeneratedclocks 0

# NFilter
set_option -popfeed 0
set_option -constprop 0
set_option -createhierarchy 0

# sequential_optimization_options
set_option -symbolic_fsm_compiler 1

# Compiler Options
set_option -compiler_compatible 0
set_option -resource_sharing 1

# automatic place and route (vendor) options
set_option -write_apr_constraint 1

# set result format/file last
project -result_format edif
project -result_file impl.edf
impl -active impl
project -run synthesis -clean
EOT

# synthesis
"$icecubedir"/sbt_backend/bin/linux/opt/synpwrap/synpwrap -prj impl_syn.prj -log impl.srr

# convert netlist
"$icecubedir"/sbt_backend/bin/linux/opt/edifparser "$icecubedir"/sbt_backend/devices/ICE40P01.dev impl/impl.edf netlist -pTQ144 -yinput.pcf -sinput.sdc -c --devicename iCE40HX1K

# run placer
"$icecubedir"/sbt_backend/bin/linux/opt/sbtplacer --des-lib netlist/oadb-top --outdir outputs/placer --device-file "$icecubedir"/sbt_backend/devices/ICE40P01.dev --package TQ144 --deviceMarketName iCE40HX1K \
	--sdc-file netlist/Temp/sbt_temp.sdc --lib-file "$icecubedir"/sbt_backend/devices/ice40HX1K.lib --effort_level std --out-sdc-file outputs/placer/top_pl.sdc

# run packer (1/2)
"$icecubedir"/sbt_backend/bin/linux/opt/packer "$icecubedir"/sbt_backend/devices/ICE40P01.dev netlist/oadb-top --package TQ144 --outdir outputs/packer --DRC_only \
	--translator "$icecubedir"/sbt_backend/bin/sdc_translator.tcl --src_sdc_file outputs/placer/top_pl.sdc --dst_sdc_file outputs/packer/top_pk.sdc --devicename iCE40HX1K

# run packer (2/2)
"$icecubedir"/sbt_backend/bin/linux/opt/packer "$icecubedir"/sbt_backend/devices/ICE40P01.dev netlist/oadb-top --package TQ144 --outdir outputs/packer \
	--translator "$icecubedir"/sbt_backend/bin/sdc_translator.tcl --src_sdc_file outputs/placer/top_pl.sdc --dst_sdc_file outputs/packer/top_pk.sdc --devicename iCE40HX1K

# run router
"$icecubedir"/sbt_backend/bin/linux/opt/sbrouter "$icecubedir"/sbt_backend/devices/ICE40P01.dev netlist/oadb-top "$icecubedir"/sbt_backend/devices/ice40HX1K.lib outputs/packer/top_pk.sdc \
	--outdir outputs/router --sdf_file outputs/simulation_netlist/top_sbt.sdf --pin_permutation

# run netlister
"$icecubedir"/sbt_backend/bin/linux/opt/netlister --verilog outputs/simulation_netlist/top_sbt.v --vhdl outputs/simulation_netlist/top_sbt.vhd --lib netlist/oadb-top --view rt \
	--device "$icecubedir"/sbt_backend/devices/ICE40P01.dev --splitio --in-sdc-file outputs/packer/top_pk.sdc --out-sdc-file netlister/top_sbt.sdc

# run timer
"$icecubedir"/sbt_backend/bin/linux/opt/sbtimer --des-lib netlist/oadb-top --lib-file "$icecubedir"/sbt_backend/devices/ice40HX1K.lib --sdc-file outputs/netlister/top_sbt.sdc \
	--sdf-file outputs/simulation_netlist/top_sbt.sdf --report-file outputs/timer/top_timing.rpt --device-file "$icecubedir"/sbt_backend/devices/ICE40P01.dev --timing-summary

# make bitmap
"$icecubedir"/sbt_backend/bin/linux/opt/bitmap "$icecubedir"/sbt_backend/devices/ICE40P01.dev --design netlist/oadb-top --device_name iCE40HX1K --package TQ144 --outdir outputs/bitmap \
	--low_power on --init_ram on --init_ram_bank 1111 --frequency low --warm_boot on
)

cp -v "$1.tmp"/outputs/bitmap/top_bitmap.bin "$1.bin"

