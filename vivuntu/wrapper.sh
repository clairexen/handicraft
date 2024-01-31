#!/bin/bash

web_command="chromium-browser"
pdf_command="okular"

function open_web() {
	echo "+ WEB <$1>" >&2
	( set -x; $web_command "$1"; ) &
	exit 0
}

function open_pdf() {
	echo "+ PDF <$1>" >&2
	( set -x; $pdf_command "$1"; ) &
	exit 0
}

echo "-- Vivuntu wrapper --" >&2
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
unset LD_LIBRARY_PATH

prog=$(basename $0)
echo "+ $prog $*" >&2

case "$prog" in
	xdg-open|chromium-browser)
		open_web "$*" ;;
	kfmclient)
		if [ "$1" = openURL ]; then open_web "$2"; fi ;;
	firefox)
		open_web "$( echo "$*" | sed 's,-remote \+openurl(,,; s,),,;' )" ;;
	acroread)
		open_pdf "$1" ;;
esac

echo "+ Vivuntu wrapper failed. Don't know what to do." >&2
exit 1

