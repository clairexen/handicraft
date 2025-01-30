#!/usr/bin/env python3
#
# The New York Times "WordleBot" is behnd a paywall.  :/
# So I wrote my own "WordleDroid" which I can run locally.
#
# Copyright (C) 2025  Claire Xenia Wolf <claire@clairexen.net>
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import json, nltk
from nltk.corpus import wordnet, words

nltk.download('words')
all_en_words = set(words.words('en'))

nltk.download('wordnet')
all_wordnet_lemmas = set(wordnet.all_lemma_names())

if True:
    dictfile = "/usr/share/dict/american-english-large"
else:
    dictfile = "/usr/share/dict/american-english-insane"

with open(dictfile) as f:
    all_dict_words = {w.strip() for w in f}

with open("wordledb.json") as f:
    all_wordle_words = {record["solution"]
                        for record in json.load(f).values()}

selected_words = set(words.words('en-basic'))

if True:
    # select the words presents in two of our three DBs
    selected_words |= all_en_words.intersection(all_wordnet_lemmas)
    selected_words |= all_en_words.intersection(all_dict_words)
    selected_words |= all_wordnet_lemmas.intersection(all_dict_words)

else:
    # select all words
    selected_words |= all_en_words
    selected_words |= all_wordnet_lemmas
    selected_words |= all_dict_words

blacklist = set("mmmm oooo xxxix".split())
selected_words.difference_update(blacklist)

missing_wordle_words = sorted(all_wordle_words.difference(selected_words))
print(f"Found {len(missing_wordle_words)}/{len(all_wordle_words)} additional wordle " +
      f"words not already selected:\n  {missing_wordle_words}")

selected_words |= all_wordle_words

with open("words.cc", "w") as f:
    for N in (4, 5, 6):
        words = "".join(sorted([w for w in selected_words if len(w) == N and
                                w.isalpha() and w.islower() and w.isascii()]))
        print(f"extern const char WordleDroidWords{N}[];", file=f)
        print(f"const char WordleDroidWords{N}[] = // {len(words)//N} words", end="", file=f)
        for k in range(0, len(words), 60):
            print(f"\n\"{words[k:k+60]}\"", end="", file=f)
        print(";", file=f)
