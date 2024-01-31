#!/bin/bash

partname="xc6slx9-2-tqg144"
xilinxdir_default="/opt/Xilinx/14.5/ISE_DS/ISE"

bitfile=""
ndffile=""
xilinxdir="$xilinxdir_default"
topmod="top"
edif2ngd_a=""
yosys=false
keep=false

help()
{
	echo >&2 ""
	echo >&2 "Usage: $0 [-k] [-a] [-t <top>] [-b <bitfile>] [-n <ndffile>] [-x <xilinx_dir>] <soures>"
	echo >&2 ""
	echo >&2 "   -k"
	echo >&2 "      keep temporary directory (for debugging)"
	echo >&2 ""
	echo >&2 "   -a"
	echo >&2 "      add PAD's to all top level port signal in first EDIF input"
	echo >&2 ""
	echo >&2 "   -y"
	echo >&2 "      use Yosys instead of XST for synthesis"
	echo >&2 ""
	echo >&2 "   -t <top>"
	echo >&2 "      top module name (default: top)"
	echo >&2 ""
	echo >&2 "   -b <bitfile>"
	echo >&2 "      output file name (default: first source file"
	echo >&2 "      with .bit suffix instead of .v or .vhd)"
	echo >&2 ""
	echo >&2 "   -n <ndffile>"
	echo >&2 "      output edif file (for further analysis)"
	echo >&2 ""
	echo >&2 "   -x <xilinx_dir>"
	echo >&2 "      path to xilinx ise installation"
	echo >&2 "      default: $xilinxdir_default"
	echo >&2 ""
	echo >&2 "   Source file extensions supported:"
	echo >&2 "      .v     Verilog (IEEE 1364-2001)"
	echo >&2 "      .vhd   VHDL (IEEE 1076-1993)"
	echo >&2 "      .edn   EDIF version 2.0.0 netlist"
	echo >&2 "      .ucf   Xilinx User Constraints File"
	echo >&2 ""
}

while getopts kayt:b:n:s: opt; do
	case "$opt" in
		k)
			keep=true ;;
		a)
			edif2ngd_a="-a" ;;
		y)
			yosys=true ;;
		t)
			topmod="$OPTARG" ;;
		b)
			bitfile="$OPTARG" ;;
		n)
			ndffile="$OPTARG" ;;
		x)
			xilinxdir="$OPTARG" ;;
		*)
			help
			exit 1
	esac
done

shift $(($OPTIND - 1))

if [ $# -eq 0 ]; then
	help
	exit 1
fi

if [ -z "$bitfile" ]; then
	bitfile="$1"
	bitfile="${bitfile%.v}"
	bitfile="${bitfile%.vhd}"
	bitfile="${bitfile%.edn}"
	bitfile="${bitfile%.ucf}"
	bitfile="$bitfile.bit"
fi
if [[ "$bitfile" != /* ]]; then
	bitfile="$PWD/$bitfile"
fi
if [ -n "$ndffile" ] && [[ "$ndffile" != /* ]]; then
	ndffile="$PWD/$ndffile"
fi

case "$(uname -m)" in
	*64) xilinxbin="$xilinxdir/bin/lin64" ;;
	*) xilinxbin="$xilinxdir/bin/lin" ;;
esac

declare -a sources=( "$@" )

set -e
tmpdir="$( mktemp -d $PWD/mojo_synth_tmp.XXXXXXXXXX )"
if ! $keep; then trap 'rm -rf "$tmpdir"' 0; fi

cat > "$tmpdir"/mojo_synth.xst << EOT
run
-ifn mojo_synth.prj
-ofn mojo_synth
-ofmt NGC
-p $partname
-top $topmod
-opt_mode Speed
-opt_level 1
-power NO
-iuc NO
-keep_hierarchy No
-netlist_hierarchy As_Optimized
-rtlview Yes
-glob_opt AllClockNets
-read_cores YES
-write_timing_constraints NO
-cross_clock_analysis NO
-hierarchy_separator /
-bus_delimiter <>
-case Maintain
-slice_utilization_ratio 100
-bram_utilization_ratio 100
-dsp_utilization_ratio 100
-lc Auto
-reduce_control_sets Auto
-fsm_extract YES -fsm_encoding Auto
-safe_implementation No
-fsm_style LUT
-ram_extract Yes
-ram_style Auto
-rom_extract Yes
-shreg_extract YES
-rom_style Auto
-auto_bram_packing NO
-resource_sharing YES
-async_to_sync NO
-shreg_min_size 2
-use_dsp48 Auto
-iobuf YES
-max_fanout 100000
-bufg 16
-register_duplication YES
-register_balancing No
-optimize_primitives NO
-use_clock_enable Auto
-use_sync_set Auto
-use_sync_reset Auto
-iob Auto
-equivalent_register_removal YES
-slice_utilization_ratio_maxmargin 5
EOT

cat > "$tmpdir"/mojo_synth.ut << EOT
-w
-g DebugBitstream:No
-g Binary:no
-g CRC:Enable
-g Reset_on_err:No
-g ConfigRate:2
-g ProgPin:PullUp
-g TckPin:PullUp
-g TdiPin:PullUp
-g TdoPin:PullUp
-g TmsPin:PullUp
-g UnusedPin:PullDown
-g UserID:0xFFFFFFFF
-g ExtMasterCclk_en:No
-g SPI_buswidth:1
-g TIMER_CFG:0xFFFF
-g multipin_wakeup:No
-g StartUpClk:CClk
-g DONE_cycle:4
-g GTS_cycle:5
-g GWE_cycle:6
-g LCK_cycle:NoWait
-g Security:None
-g DonePipe:No
-g DriveDone:No
-g en_sw_gsr:No
-g drive_awake:No
-g sw_clk:Startupclk
-g sw_gwe_cycle:5
-g sw_gts_cycle:4
EOT

declare -a ngdbuild_uc_opts=()
declare -a trce_ucf_opts=()
declare -a edif_inputs=()
declare -a ngdbuild_inputs=()
declare -a yosys_inputs=()
echo -n > "$tmpdir"/mojo_synth.prj

for file in "${sources[@]}"; do
	if [[ "$file" != /* ]]; then
		file="$PWD/$file"
	fi
	if [[ "$file" == *.v ]]; then
		if $yosys; then
			yosys_inputs=( "${yosys_inputs[@]}" "$file" )
		else
			echo "verilog work \"$file\"" >> "$tmpdir"/mojo_synth.prj
		fi
	elif [[ "$file" == *.vhd ]]; then
		echo "vhdl work \"$file\"" >> "$tmpdir"/mojo_synth.prj
	elif [[ "$file" == *.edn ]]; then
		edif_inputs=( "${edif_inputs[@]}" "$file" )
	elif [[ "$file" == *.ucf ]]; then
		ngdbuild_uc_opts=( "${ngdbuild_uc_opts[@]}" -uc "$file" )
		trce_ucf_opts=( "${trce_ucf_opts[@]}" -ucf "$file" )
	else
		echo >&2 "Unrecogniced file extension (only .v, .vhd and .ucf supported): $file"
		exit 1
	fi
done > "$tmpdir"/mojo_synth.prj

cd "$tmpdir"
if [ -s mojo_synth.prj ]; then
	$xilinxbin/xst -ifn mojo_synth.xst
	if [ -n "$ndffile" ]; then
		$xilinxbin/ngc2edif -w mojo_synth.ngc "$ndffile"
	fi
	ngdbuild_inputs=( "${ngdbuild_inputs[@]}" mojo_synth.ngc )
fi
if $yosys; then
	yosys -p "synth_xilinx -top $topmod -edif yosys_out.edif" "${yosys_inputs[@]}"
	edif_inputs=( "${edif_inputs[@]}" yosys_out.edif )
fi
for edn in "${edif_inputs[@]}"; do
	ngo_file=mojo_edn${#ngdbuild_inputs[@]}.ngo
	$xilinxbin/edif2ngd $edif2ngd_a -p $partname "$edn" $ngo_file
	ngdbuild_inputs=( "${ngdbuild_inputs[@]}" $ngo_file )
	edif2ngd_a=""
done
$xilinxbin/ngdbuild "${ngdbuild_uc_opts[@]}" -p $partname "${ngdbuild_inputs[@]}" mojo_synth.ngd
$xilinxbin/map -p $partname -w -o mojo_mapped.ncd mojo_synth prffile.pcf
$xilinxbin/par -w mojo_mapped.ncd mojo_synth.ncd prffile.pcf
$xilinxbin/trce -v -s 2 -n 3 mojo_synth.ncd prffile.pcf "${trce_ucf_opts[@]}"
$xilinxbin/bitgen -f mojo_synth.ut mojo_synth.ncd 10> webtalk.pid

# kill webtalk now (it's automatically launched in background by bitgen)
# (otherwise we get an ungly error message because we removed its working dir)
fuser -k webtalk.pid

cp -v mojo_synth.bit "$bitfile"
echo READY.

