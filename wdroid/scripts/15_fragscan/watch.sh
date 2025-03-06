# Usage:
#   watch -n60 bash watch.sh

grep -h '^Maximum depth at optimal play: ' *.out | sort -V | uniq -c
echo

echo total $(
grep -Ph '^Maximum depth at optimal play: ([7-9]|[1-9][0-9])' *.out | wc -l
) '"bad" first guesses (with depth >= 7)'
echo

echo $(
grep -L '^Maximum depth at optimal play: ' *.out | wc -l
) output files without results, $(
grep -L 'user .*system .*elapsed' *.out | wc -l
) unfinished
