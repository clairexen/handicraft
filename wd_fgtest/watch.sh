# Usage: watch bash watch.sh
grep -Pho '(?<=^Maximum depth at optimal play: )\d+' *.out | \
	sort | uniq -c | sed 's/^ *[0-9]*/&x max depth:/; s/$/ guesses/;'
echo; echo `grep -L '^Maximum depth at optimal play: ' *.out | wc -l` \
	failed jobs: `grep -L '^Maximum depth at optimal play: ' *.out | cut -c5-9`
echo; tail -n8 *.new
