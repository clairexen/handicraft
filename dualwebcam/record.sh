
avconv_record() {
	cp empty.png $3.png
	./libav/avconv -y -f video4linux2 -r 12 -s $2 -i $1 -filter_complex '[0:v]fps=25[vo]' \
			-map '[vo]' -c:v libx264 -preset ultrafast $3.mp4 -map 0:v -r 8 -f image2 -update 1 $3.png &
	pqiv -w -P off $3.png 10> $3.pid &
}

{
# arecord -f s16 -c1 -r44000 -D plughw:U0x46d0x81b webcam_audio.wav &
avconv_record /dev/video0 1280x720 webcam_angle0
avconv_record /dev/video1  640x360 webcam_angle1
sleep 1; fuser webcam_pids.txt >&10
} 10> webcam_pids.txt

read
fuser -INT -k webcam_pids.txt
fuser -KILL -k webcam_angle0.pid webcam_angle1.pid

echo "Waiting for processes to finish..."
wait

rm -f webcam_pids.txt
rm -f webcam_angle0.pid webcam_angle1.pid
rm -f webcam_angle0.png webcam_angle1.png

