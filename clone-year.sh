#!/bin/bash
for year; do
	case "$year" in
		200[89]|201[0-9]|202[01234])
			if test -d "handicraft-$year"; then
				echo "directory handicraft-$year already exists"
				continue
			fi
			git clone --single-branch --no-tags --reference "$PWD" -b "handicraft-$year" \
					"git@github.com:clairexen/handicraft.git"  handicraft-"$year" ;;
		*)
			echo "eror processing arg $year: arg must be a year within the valid range"
			exit 1 ;;
	esac
done
