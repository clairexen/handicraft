#!/bin/bash

id="$1"
title="$2"

EMAIL="example@example.org"
PASS="secret"

MEDIA_UUID="213C-0260"
MEDIA_MOUNT="/mnt/ati"
MEDIA_PATH="DCIM/100EKZM1"
TRAILER="trailer.avi"
LEDTTS="/dev/tts/0"

if [ -f settings.sh ]; then
	. settings.sh
fi

function v() {
	echo "+ $*"
	"$@"
}

echo "Running: process.sh '$id' '$title'"
date

v mencoder -quiet -oac mp3lame -ovc lavc -ofps 24 -vf scale=640:480 -o videos/$id/_final.avi $TRAILER videos/$id/[!_]*.avi

for x in 1 2 3 4 5; do
	v youtube-upload --wait-processing $EMAIL $PASS videos/$id/_final.avi "$title" "Automatic upload from Metalab's \"Alert The Internets\" station:
http://metalab.at/wiki/Alert_The_Internets" "Nonprofit" "metalab" && break
	if [ $x -eq 5 ]; then
		echo "The internet has been deleted! giving up.."
		break
	fi
	echo "Error from youtube. Re-try in one minute.."
	sleep 60
done

rm -vf videos/$id/_final.avi

date

