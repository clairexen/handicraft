#!/bin/bash

set -ex

rm -rf tmp/$1
mkdir -p tmp/$1
cd tmp/$1

wget http://ftp.ports.debian.org/debian-ports/$2
ar x $(basename $2)

if test -f data.tar.xz; then
	tar xvJf data.tar.xz
elif test -f data.tar.gz; then
	tar xvzf data.tar.gz
else
	false
fi

bash ../../analyze.sh > ../../database/$1.new
mv ../../database/$1.new ../../database/$1.txt

cd ..
rm -rf $1
