from datasets import load_dataset
from itertools import islice
from pprint import pprint
import random, sys, gzip

MIN_LINE = 200
MAX_LINE = 1024
TEST_SPLIT = 0.11
SEP = "\n<|----|>\n"

WP_LANG = sys.argv[1]  # e.g. 'simple' or 'en'
if len(sys.argv) == 4:
    NUM_SHARDS = int(sys.argv[2])
    SHARD_IDX = int(sys.argv[3])
    OUT_TRAIN = f"wp-{WP_LANG}-{SHARD_IDX:04d}-train.txt.gz"
    OUT_TEST = f"wp-{WP_LANG}-{SHARD_IDX:04d}-test.txt.gz"
else:
    NUM_SHARDS = None
    SHARD_IDX = None
    OUT_TRAIN = f"wp-{WP_LANG}-train.txt.gz"
    OUT_TEST = f"wp-{WP_LANG}-test.txt.gz"

#random.seed()    # nondeterministic
random.seed(1234) # set a seed here if you want reproducibility

# ---------------------------------------------------

#from datasets import get_dataset_config_names
#print(get_dataset_config_names("wikimedia/wikipedia"))

dataset = load_dataset(
     "wikimedia/wikipedia",
     f"20231101.{WP_LANG}",
     split="train"
)

if NUM_SHARDS:
    dataset = dataset.shard(num_shards=NUM_SHARDS, index=SHARD_IDX)

test = []
train = []

def split_center_cond(text):
    if len(text) < MAX_LINE:
        return [text]
    splits = [i for i in range(len(text)) if text.startswith(". ", i)]
    if not splits:
        #print(f"rejecting overlong sentence of length {len(text)}.")
        #pprint(text)
        return []
    center = len(text) // 2
    i = min(splits, key=lambda x: abs(x - center))
    return split_center_cond(text[:i+1]) + split_center_cond(text[i+2:])

def process_row(row, is_test):
    for text in row['text'].lower().split("\n"):
        if len(text) < MIN_LINE:
            continue
        snippets = split_center_cond(text)
        if is_test:
            test.extend(snippets)
        else:
            train.extend(snippets)

print(f"Processing...")
for i, row in enumerate(dataset):
    if i and not i % 100000:
        print(f"<processed {i} rows ({100*i/len(dataset):.2f}%) so far>")
    process_row(row, random.random() < TEST_SPLIT)

# ---------------------------------------------------

print(f"test records:  {len(test)}", file=sys.stderr)
print(f"train records: {len(train)}", file=sys.stderr)

print(f"Shuffe: test records")
random.shuffle(test)

print(f"Write: {OUT_TEST}")
with gzip.open(OUT_TEST, "wt", encoding="utf-8") as f:
    for t in test:
        f.write(t.strip())
        f.write(SEP)

print(f"Shuffe: train records")
random.shuffle(train)

print(f"Write: {OUT_TRAIN}")
with gzip.open(OUT_TRAIN, "wt", encoding="utf-8") as f:
    for t in train:
        f.write(t.strip())
        f.write(SEP)
