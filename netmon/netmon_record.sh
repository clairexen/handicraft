#!/bin/bash
inet_ip=173.194.70.102
while true
do
	sleep 10 &
	ping -c3 -W5 $inet_ip > /dev/null 2>&1
	status=$?
	wait
	date "+%s $(( !status ))"
done
