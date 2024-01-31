#!/bin/bash

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

sudo umount $MEDIA_MOUNT 2> /dev/null
sudo umount $MEDIA_MOUNT 2> /dev/null
sudo umount $MEDIA_MOUNT 2> /dev/null

sudo rm -f ledctrl.cmd
sudo fuser -k ./ledctrl
sudo ./ledctrl $LEDTTS &
trap "echo x > ledctrl.cmd" 0

set -e
echo . > ledctrl.cmd
echo; echo; echo
echo "Waiting for media..."
while sleep 1
do
	if dev=$(blkid blkid -l -o device -t UUID=$MEDIA_UUID) && [ "$dev" != "" ]
	then
		echo r > ledctrl.cmd
		read id title < <( date "+%Y%m%d%H%M%S Alert The Internets %Y-%m-%d %H:%M:%S"; )
		echo "Found media as $dev. New video ID is $id."

		echo -n "Copying data..."
		sudo mount -o uid=$UID $dev $MEDIA_MOUNT
		if ls $MEDIA_MOUNT/$MEDIA_PATH/*.[Aa][Vv][Ii] > /dev/null 2>&1
		then
			mkdir -p videos/$id
			mv $MEDIA_MOUNT/$MEDIA_PATH/*.[Aa][Vv][Ii] videos/$id/
			sudo umount $MEDIA_MOUNT
			echo " done."

			echo "Starting background convertion and upload of video file to youtube."
			nohup bash process.sh "$id" "$title" > videos/$id/_process.log 2>&1 &
		else
			sudo umount $MEDIA_MOUNT
			echo " nothing there -> nothing to do."
			sleep 2
		fi

		echo g > ledctrl.cmd
		echo -n "Waiting for media to disconnect again..."
		while [ -b $dev ]; do sleep 1; done
		echo " done."

		echo . > ledctrl.cmd
		echo; echo; echo
		echo "Waiting for media..."
	fi
done

