#!/bin/bash

exec > /dev/null 2>&1
timestamp=$( date '+%Y%m%d%H%M' )
videodevice=/dev/video0

# record a short video to let the auto-exposure of the webcam adjust
avconv -y -f video4linux2 -r 7 -s 1024x768 -i $videodevice -t 60 -c:v libx264 -preset ultrafast vid_$timestamp.mp4

# record a series aof ppm files
avconv -y -f video4linux2 -r 7 -s 1024x768 -i $videodevice -frames 100 '_temp_%03d.ppm'

# consolidate the ppm files by calculating the median of each rgb value
clang -Wall -o medianbytes medianbytes.c
./medianbytes _temp_*.ppm > shot_$timestamp.ppm

# clean up temp files
rm -f _temp_*.ppm vid_$timestamp.mp4

