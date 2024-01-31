#!/bin/bash

set -ex

./avsync_audio handycam.mp4 | tee handycam.mrk
./avsync_video webcam_angle0.mp4 | tee webcam_angle0.mrk
./avsync_video webcam_angle1.mp4 | tee webcam_angle1.mrk

cat handycam.mrk webcam_angle0.mrk webcam_angle1.mrk | python avsync_analyse.py

