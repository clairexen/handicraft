#!/bin/bash

v() { echo "+ $*"; "$@"; }

v svn pd svn:executeable */*.jpg */*.png
v svn ps svn:mime-type image/png */*.png
v svn ps svn:mime-type image/jpeg */*.jpg

for in_file in *.in; do
	v python genhtml.py ${in_file} ${in_file%.in}.html
	v svn add ${in_file%.in}.html
	v svn ps svn:mime-type text/html ${in_file%.in}.html
done

if [ -f localmake.sh ]; then
	. localmake.sh
fi

