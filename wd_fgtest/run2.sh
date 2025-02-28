words="block buxom fjord jinks jumbo pluck squib zombi" # "failed" jobs from run.sh
words="`echo $words | tr ' ' '\n' | sort -R`"
{
	echo -n "all:"
	for word in $words; do
		echo -n " tst_$word.out2"
	done; echo
	for word in $words; do
		echo "tst_$word.out2:"
		echo "	-time wdroid -rlimitDataGBs=28 -rlimitCpuMins=30 \\"
		echo "		-minmax +f=$word +go -quit > tst_$word.new2 2>&1"
		echo "	tail -n15 tst_$word.new2"
		echo "	[ \`stat -c%s tst_$word.new2\` -gt 2000 ]"
		echo "	mv -v tst_$word.new2 tst_$word.out2"
	done
} > run2.mk
rm -f tst_*.new2
make -j1 -f run2.mk
