#!/bin/bash

set -e

make -C libmetaleds libmetaleds.o

cd mplayer-sources

disable_orgy=$( ./configure --help | egrep -- '--(enable|disable).*(output)' | awk '{ print $1; }' | sed 's,enable,disable,' | egrep -v 'metaleds|alsa|ossaudio'; )
CFLAGS= ./configure $disable_orgy --enable-metaleds

make

cp mplayer ../mplayer-bin

