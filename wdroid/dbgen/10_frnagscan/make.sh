# Usage:
#   bash make.sh

N=5      # number of letters in word
M=30     # number of minutes CPU time per job
G=16     # number of GB ram per job
J=2      # number of parallel make jobs
P=false  # prune output files without result line

for opt; do
	case "$opt" in
		-N*) N=${opt#-?} ;;
		-M*) M=${opt#-?} ;;
		-G*) G=${opt#-?} ;;
		-J*) J=${opt#-?} ;;
		-P) P=true ;;
	esac
done

mkdir -p data$N

if $P; then
	grep -L '^Maximum depth at optimal play:' data$N/*.out | xargs -r rm -v
fi

{
	echo -n "all:"
	sort -R ../../words$N.txt | while read word; do echo -n " $word.out"; done
	echo
	while read word; do
		echo "$word.out:"
		echo "	-time ../../../wdroid -$N -rlimitCpuMins=$M -rlimitDataGBs=$G -minmax +frnag `
				`+f=$word +maxTraceLength=7 +go -quit > $word.out 2>&1"
	done < ../../words$N.txt
} > data$N/run.mk

make -C data$N -j $J -f run.mk
grep -l '^Maximum depth at optimal play: [1-6]$' data$N/*.out | cut -f1 -d. > data$N/goodfg$N.txt
grep -Pl '^Maximum depth at optimal play: ([7-9]|[1-9][0-9])' data$N/*.out | cut -f1 -d. > data$N/badfg$N.txt
