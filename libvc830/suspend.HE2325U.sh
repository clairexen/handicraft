#!/bin/bash
# see http://www.erste.de/UT61/index.html
for dat in /sys/bus/usb/devices/*; do 
	if test -e $dat/manufacturer && grep -q "WCH.CN" $dat/manufacturer; then
		echo "Suspending ${dat}."
		echo auto > ${dat}/power/control
		echo 0 > ${dat}/power/autosuspend 
	fi      
done
