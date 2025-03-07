# Usage:
#   watch -n60 bash watch.sh -N5

N=5

for opt; do
	case "$opt" in
		-N*) N=${opt#-?} ;;
	esac
done

grep -h '^Maximum depth at optimal play: ' data$N/*.out | sort -V | uniq -c
echo

echo total $(
grep -Ph '^Maximum depth at optimal play: ([7-9]|[1-9][0-9])' data$N/*.out | wc -l
) '"bad" first guesses (with depth >= 7)'
echo

echo $(
grep -L '^Maximum depth at optimal play: ' data$N/*.out | wc -l
) output files without results, $(
grep -L 'user .*system .*elapsed' data$N/*.out | wc -l
) unfinished
