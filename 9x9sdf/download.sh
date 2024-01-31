#!/bin/bash

set -ex

download() {
	idxfile=index_"$( echo "$1" | tr -d :/ )".html
	[ -f ${idxfile} ] || { wget -O ${idxfile}.part "$1" && mv ${idxfile}.part ${idxfile}; }
	for x in $( tr '"' '\n' < ${idxfile} | grep -v '?' | grep '\.sgf$'; ); do
		f=$( echo "$x" | tr -d /; ); u="http://gobase.org$x"
		[ -f "$f" ] || { wget -O "${f}.part" "$u" && mv "${f}.part" "${f}"; }
	done
}

download http://gobase.org/9x9/goseigen/
download http://gobase.org/9x9/book/part1/
download http://gobase.org/9x9/book/part2/
download http://gobase.org/9x9/book/part3/
download http://gobase.org/9x9/book4/
download http://gobase.org/9x9/magazine/1992/
download http://gobase.org/9x9/magazine/1993/
download http://gobase.org/9x9/tournament/
download http://gobase.org/9x9/japan/female/2002/

