words="abort above abysm aleph ambit ampul amuck asdic aspic bedim befit befog
       begum bench biped blitz block blush bogus brick brink brisk bunch buxom
       cabin champ chomp clamp clasp climb cloak cramp crisp cupel curve debit
       defog demob depot dregs faced fecal felid filmy final fixed fjord flash
       flask flesh flirt flush glove grosz grump hemic hoard hovel humid imbed
       impel jinks jumbo leapt limbo links lymph medal misdo mufti plash plasm
       plica pluck plush prick prink prism psalm pubic pubis pucka rhomb scald
       scalp scamp scorn scrip scrub scrum scurf sebum sepal shard shawm shelf
       shrub sirup skimp slink slurp smack smart smirk smoke spawn spick spirt
       spoil sprag sprit squab squib strip sulfa swamp tepid tramp trump vapid
       velum venom wench zombi"
words="`echo $words | tr ' ' '\n' | sort -R`"
{
	echo -n "all:"
	for word in $words; do
		echo -n " tst_$word.out"
	done; echo
	for word in $words; do
		echo "tst_$word.out:"
		echo "	-time wdroid -rlimitDataGBs=16 -rlimitCpuMins=15 \\"
		echo "		-minmax +f=$word +go -quit > tst_$word.new 2>&1"
		echo "	tail -n15 tst_$word.new"
		echo "	[ \`stat -c%s tst_$word.new\` -gt 2000 ]"
		echo "	mv -v tst_$word.new tst_$word.out"
	done
} > run.mk
rm -f tst_*.new
make -j2 -f run.mk
