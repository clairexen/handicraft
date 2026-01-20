#!/bin/bash
set -ve

python shrink.py --corpus simplewiki
mv shrink-train.png data/simplerwiki-train.png
gzip --rsyncable -9 < shrink-train.txt > data/simplerwiki-train.txt.gz
for x in 1 2 3; do python shrink.py; done
gzip --rsyncable -9 < shrink-train.txt > data/simplestwiki-train.txt.gz

python shrink.py --corpus simplewiki --test
mv shrink-test.png data/simplerwiki-test.png
gzip --rsyncable -9 < shrink-test.txt > data/simplerwiki-test.txt.gz
for x in 1 2 3; do python shrink.py; done
gzip --rsyncable -9 < shrink-test.txt > data/simplestwiki-test.txt.gz

rm -f data/simple{,r,st}wiki_tokens_test_2000.pt
rm -f shrink-{train,test}.{txt,png}
exit 0
