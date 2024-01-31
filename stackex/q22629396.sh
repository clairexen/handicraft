#!/bin/bash
# http://stackoverflow.com/questions/22629396/cant-get-git-to-correctly-handle-crlf-problems-with-eclipse-launch-configs-xml

set -ex

rm -rf q22629396
mkdir q22629396
cd q22629396

git init

echo '*.txt text' > .gitattributes
git add .gitattributes
git commit -m 'added .gitattributes'

echo -e 'line with LF' > test.txt
echo -e 'line with CRLF\r' >> test.txt
echo -e 'line with LF' >> test.txt
echo -e 'line with CRLF\r' >> test.txt

mkdir subdir
echo -e 'line with LF' > subdir/test.txt
echo -e 'line with CRLF\r' >> subdir/test.txt
echo -e 'line with LF' >> subdir/test.txt
echo -e 'line with CRLF\r' >> subdir/test.txt

git add test.txt subdir/test.txt
git commit -m 'added test.txt and subdir/test.txt'

git show HEAD:test.txt | hexdump -c
git show HEAD:subdir/test.txt | hexdump -c
