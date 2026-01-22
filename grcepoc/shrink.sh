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
for x in 1 2 3; do python shrink.py --test; done
gzip --rsyncable -9 < shrink-test.txt > data/simplestwiki-test.txt.gz

rm -f shrink-{train,test}.{txt,png}
rm -vf data/simple{r,st}wiki_tokens_{test,train}_2000.pt

# rm -vf data/simple{r,st}wiki_vocab_2000.json
# python grce.py corpus --init-tokenizer
# cp -v data/simple{r,st}wiki_vocab_2000.json

exit 0
