#!/bin/bash

set -ex
export

case "$DEB_BUILD_ARCH-$DEB_HOST_ARCH" in
	amd64-armhf)
		make CC=arm-linux-gnueabihf-gcc
		;;
	*)
		make
		;;
esac
