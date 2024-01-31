#!/bin/bash

CFG_DIST="trusty"
CFG_MIRROR="http://ports.ubuntu.com/"
CFG_PROXY="http://192.168.0.1:3128"

if ! test -w /; then
	echo "This script must be run as root!" >&2
	exit 1
fi

set -x
for d in dev sys proc; do umount -vl $PWD/rootfs/$d 2> /dev/null; done
rm -rf Tegra124_Linux_R21.2.0_armhf.tbz2 Linux_for_Tegra rootfs
export ftp_proxy="$CFG_PROXY" http_proxy="$CFG_PROXY" https_proxy="$CFG_PROXY"

# install basic root fs
debootstrap --foreign --arch=armhf $CFG_DIST rootfs $CFG_MIRROR

# second stage: setup
cd rootfs
cp -v `type -p qemu-arm-static` ./`type -p qemu-arm-static`
for d in dev sys proc; do mount --bind /$d $PWD/$d; mount -o remount,ro $PWD/$d; done

# second stage: install
{
	echo "export ftp_proxy='$CFG_PROXY' http_proxy='$CFG_PROXY' https_proxy='$CFG_PROXY'"
	echo "./debootstrap/debootstrap --second-stage" 
	for x in $CFG_DIST $CFG_DIST-security $CFG_DIST-updates; do
		x="`printf '%-15s' $x`"
		echo "echo \"deb     $CFG_MIRROR $x main restricted universe multiverse\" >> etc/apt/sources.list"
		echo "echo \"deb-src $CFG_MIRROR $x main restricted universe multiverse\" >> etc/apt/sources.list"
	done
	echo "apt-get update"
	# echo "apt-get -y install ubuntu-desktop"
} > chroot-run.sh
chroot . bash -x chroot-run.sh

# second stage: cleanup
rm ./`type -p qemu-arm-static` chroot-run.sh
for d in dev sys proc; do umount -vl $PWD/$d 2> /dev/null; done

# install linux4tegra drivers
cd ..
wget http://developer.download.nvidia.com/mobile/tegra/l4t/r21.2.0/pm375_release_armhf/Tegra124_Linux_R21.2.0_armhf.tbz2
tar xvjf Tegra124_Linux_R21.2.0_armhf.tbz2
cd Linux_for_Tegra
rm -rf rootfs; ln -s ../rootfs .
./apply_binaries.sh

