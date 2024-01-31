#!/bin/bash

list_nop=""
list_cover=""
list_miss=""
list_strange=""

for i in {0..1000}; do
	[ -f mut_simout_${i}.txt ] || break
	cmp -s mut_simout_0.txt mut_simout_${i}.txt
	diff_sim=$?

	grep -q PASS mut_equiv_${i}/status
	diff_equiv=$?

	case "$diff_sim$diff_equiv" in
		00) list_nop="$list_nop $i" ;;
		11) list_cover="$list_cover $i" ;;
		01) list_miss="$list_miss $i" ;;
		10) list_strange="$list_strange $i" ;;
	esac
done

pr() {
	echo
	echo "$2 (`echo $1 | wc -w`):"
	for i in $1; do
		egrep -- "-ctrl [a-z]+ [0-9]+ $i " picorv32.mut
	done
}

pr "$list_nop" "Apparent non-mutations"
pr "$list_cover" "Covered mutations"
pr "$list_miss" "Missed (not covered) mutations"
pr "$list_strange" "Covered mutations that pass formal"
echo
