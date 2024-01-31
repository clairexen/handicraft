#!/bin/bash

i=0
rm -f _video_*.ppm

for f in $( ls shot_*.ppm ); do
	for k in 1 2 3 4 5; do
		ln -s $f $( printf "_video_%05d.ppm" $i )
		i=$( expr $i + 1 )
	done
done

avconv -y -r 25 -i _video_%05d.ppm -c:v libx264 -preset fast video.mp4
rm -f _video_*.ppm

