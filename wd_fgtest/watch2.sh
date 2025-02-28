# Usage: watch bash watch.sh
echo "Run #1:"
grep -Pho '(?<=^Maximum depth at optimal play: )\d+' *.out | \
	sort | uniq -c | sed 's/^ *[0-9]*/&x max depth:/; s/$/ guesses/;'
grep -L '^Maximum depth at optimal play: ' *.out | \
	sed 's/.*/failed job/' | uniq -c | sed 's/^ *[0-9]*/&x/'
echo; echo "Run #2:"
grep -Pho '(?<=^Maximum depth at optimal play: )\d+' *.out2 | \
	sort | uniq -c | sed 's/^ *[0-9]*/&x max depth:/; s/$/ guesses/;'
echo; echo `grep -L '^Maximum depth at optimal play: ' *.out2 | wc -l` \
	failed jobs: `grep -L '^Maximum depth at optimal play: ' *.out2 | cut -c5-9`
echo; tail -n8 *.new2
