import pyphen
from collections import defaultdict

totalcount = 0
words = defaultdict(int)
def readwords(filename):
    global totalcount, words
    print(f"Reading words from {filename}...")
    with open(filename) as f:
        for line in f:
            if "'''" in line: continue
            for word in "".join((ch if ch.islower() else ' ') for ch in line.lower()).split():
                if len(word) > 1:
                    totalcount += 1
                    words[word] += 1

readwords("datasets/simplewiki-plain/simplewiki-test.asc")
readwords("datasets/simplewiki-plain/simplewiki-train.asc")
print(f"Word count: {totalcount} (dup), {len(words)} (dedup)")

print("Converting words to token database...")

# English hyphenation dictionary
dic = pyphen.Pyphen(lang='en')

tokens = defaultdict(int)
for word, cnt in words.items():
    for tok in dic.inserted(word).split("-"):
        tokens[tok] += cnt
    tokens[word] += cnt

numtokens = 8000
sorted_tokens = sorted((-v,k) for k,v in tokens.items())[:numtokens]
print("Tokens:", sorted_tokens[:100], "...", sorted_tokens[-10:])

print("Writing final token list to lextokens.txt.")
with open("lextokens.txt", "w") as f:
    for _, tok in sorted_tokens: print(tok, file=f)
