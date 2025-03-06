# Usage:
#   bash run.sh
#
# Note: Run ../10_fragscan/ to completion before running this script.

# Config
N=5 M=10 G=6 J=6
#N=5 M=30 G=16 J=2

# Run this to remove out files with no result in them:
#   grep -L '^Maximum depth at optimal play:' *.out | xargs -r rm -v
#
# This is useful for re-running failed jobs
# after increasing $M and/or $G (or decreasing $J).

{
	echo -n "all:"
	sort -R ../10_frnagscan/goodfg$N.txt | while read word; do echo -n " $word.out"; done
	echo
	while read word; do
		echo "$word.out:"
		echo "	-time ../../wdroid -$N -rlimitCpuMins=$M -rlimitDataGBs=$G -minmax +frag `
				`+f=$word +maxTraceLength=7 +go -quit > $word.out 2>&1"
	done < ../10_frnagscan/goodfg$N.txt
} > run.mk

make -j$J -f run.mk
grep -l '^Maximum depth at optimal play: [1-6]$' *.out | cut -f1 -d. > goodfg$N.txt
grep -Pl '^Maximum depth at optimal play: ([7-9]|[1-9][0-9])' *.out | cut -f1 -d. > badfg$N.txt
