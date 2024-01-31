#!/bin/bash
#
#  Convert Eagle .brd files to gcode .ngc files for milling PCBs
#
#  Copyright (C) 2010  Clifford Wolf <clifford@clifford.at>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#

BRDFILE="brd2ngc.brd"
OUTDIR="brd2ngc.out"
BASEDIR="$(dirname `type -p $0` 2> /dev/null)"
MACHINE="sim"

help() {
	echo "Usage: $0 [-i brd_file] [-o output_dir] [-m machine_id]" >&2
	exit 1
}

eval "set -- $(getopt -o "i:o:m:" -n "$0" -- "$@")"
while [ $# -gt 0 ]; do
	if [ "$1" = "--" ]; then
		shift; break
	fi
	case "$1" in
		-i)
			BRDFILE="$2"; shift 2;;
		-o)
			OUTDIR="$2"; shift 2;;
		-m)
			MACHINE="$2"; shift 2;;
		*)
			help
	esac
done

if [ $# -gt 0 ]; then
	help
fi

if [ ! -f "$BRDFILE" ]; then
	{ echo "Can't open brd file '$BRDFILE'!"; echo; } >&2
	help
fi

set -e

DPI=500
EAGLE="eagle"
GERBV="gerbv"
CONVERT="convert"
BRDTRACE="${BASEDIR:-.}/brdtrace"

BRDTRACEOPTS=""
BRDTRACEOPTS_BACK=""
BRDTRACEOPTS_FRONT=""
BRDTRACEOPTS_OUTLINE=""
BRDTRACEOPTS_DRILL=""

case "$MACHINE" in
	sim)
		# Default simulation settings
		BRDTRACEOPTS="-z 2 -f 1000 -d 0.8"
		BRDTRACEOPTS_BACK="-c 1"
		BRDTRACEOPTS_FRONT="-c 1"
		BRDTRACEOPTS_OUTLINE="-c 5 -I 3"
		BRDTRACEOPTS_DRILL="-c 5 -I 3"
		;;
	geilomat)
		# Metalab Geil-O-Mat Settings
		BRDTRACEOPTS="-z 2 -f 1000 -d 0.6 -E'S1 M3 G4 P2|M7|M8|G64 P0.1' -S'G53 G0 Z0'"
		BRDTRACEOPTS_BACK=""
		BRDTRACEOPTS_FRONT=""
		BRDTRACEOPTS_OUTLINE="-c 2.5 -f 400 -I 2"
		BRDTRACEOPTS_DRILL="-c 2.5 -f 350 -I 2"
		;;
	*)
		if [ -f "$MACHINE.mach" ]; then
			. "$MACHINE.mach"
		else
			echo "Can't fine machine file \`$MACHINE.mach'!" >&2
			exit 1
		fi
esac

mkdir -p "$OUTDIR"
OUTDIR="$(cd $OUTDIR; pwd; )"

echo; echo "Exporting gerber and excellon files from eagle board file.."
$EAGLE -X -dgerber_rs274x -c- -r -o "$OUTDIR/layers_back.gerber" "$BRDFILE" 16 17 18
$EAGLE -X -dgerber_rs274x -c- -r -o "$OUTDIR/layers_front.gerber" "$BRDFILE" 1 17 18
$EAGLE -X -dgerber_rs274x -c- -r -o "$OUTDIR/layers_dimen.gerber" "$BRDFILE" 20
$EAGLE -X -dexcellon -c- -r -o "$OUTDIR/layers_drill.excellon" "$BRDFILE" 44 45

for x in back front dimen drill
do
	echo; echo "Creating $x bitmap file using gerbv and convert.."
	back_vis=f; front_vis=f; dimen_vis=f; drill_vis=t; eval "${x}_vis=t"
	if [ "$x" = dimen ]; then drill_vis=f; fi
	cat > "$OUTDIR/bitmap_$x.gvp" <<- EOT
		(gerbv-file-version! "2.0A")
		(define-layer! 1 (cons 'filename "$OUTDIR/layers_back.gerber")(cons 'visible #$back_vis)(cons 'color #(65535 65535 65535)))
		(define-layer! 2 (cons 'filename "$OUTDIR/layers_front.gerber")(cons 'visible #$front_vis)(cons 'color #(65535 65535 65535)))
		(define-layer! 3 (cons 'filename "$OUTDIR/layers_dimen.gerber")(cons 'visible #$dimen_vis)(cons 'color #(65535 65535 65535)))
		(define-layer! 4 (cons 'filename "$OUTDIR/layers_drill.excellon")(cons 'visible #$drill_vis)(cons 'color #(65535 65535 65535)))
		(define-layer! 5 (cons 'filename "$OUTDIR/layers_dimen.gerber")(cons 'visible #t)(cons 'color #(0 0 0)))
		(define-layer! -1 (cons 'filename "$OUTDIR")(cons 'visible #f)(cons 'color #(0 0 0)))
		(set-render-type! 0)
	EOT
	$GERBV --dpi=$DPI --export=png -o "$OUTDIR/bitmap_$x.png" -p "$OUTDIR/bitmap_$x.gvp"
	$CONVERT -rotate 90 "$OUTDIR/bitmap_$x.png" "$OUTDIR/bitmap_$x.pbm"

	echo; echo "Creating $x CNC path using brdtrace.."
	brd_extra_opts=""
	[ $x = back ] && brd_extra_opts="$BRDTRACEOPTS_BACK"
	[ $x = front ] && brd_extra_opts="-m $BRDTRACEOPTS_FRONT"
	[ $x = dimen ] && brd_extra_opts="-X $BRDTRACEOPTS_OUTLINE"
	[ $x = drill ] && brd_extra_opts="-D $BRDTRACEOPTS_DRILL"
	eval "$BRDTRACE -r $DPI $BRDTRACEOPTS $brd_extra_opts -i '$OUTDIR/bitmap_$x.pbm' -o '$OUTDIR/output_$x.ngc' -p '$OUTDIR/output_$x.ppm'"
	$CONVERT "$OUTDIR/output_$x.ppm" "$OUTDIR/output_$x.png"
	wc -l "$OUTDIR/output_$x.ngc"

	rm -f "$OUTDIR/bitmap_$x.png"
	rm -f "$OUTDIR/output_$x.ppm"
done

composite -compose add "$OUTDIR//output_back.png" "$OUTDIR/output_drill.png" "$OUTDIR/overview.png"
composite -compose add "$OUTDIR/overview.png" "$OUTDIR/output_dimen.png" "$OUTDIR/overview.png"

