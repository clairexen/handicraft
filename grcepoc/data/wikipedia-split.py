from datasets import load_dataset
from itertools import islice
from pprint import pprint
import random, sys, gzip

MIN_LINE = 200
MAX_LINE = 1024
TEST_SPLIT = 0.11
SEP = "\n\n<|----|>\n\n"

WP_LANG = sys.argv[1]  # e.g. 'simple' or 'en'
if len(sys.argv) == 4:
    NUM_SHARDS = int(sys.argv[2])
    SHARD_IDX = int(sys.argv[3])
    OUT_TRAIN = f"wikipedia-{WP_LANG}-{SHARD_IDX:04d}-train.txt.gz"
    OUT_TEST = f"wikipedia-{WP_LANG}-{SHARD_IDX:04d}-test.txt.gz"
else:
    NUM_SHARDS = None
    SHARD_IDX = None
    OUT_TRAIN = f"wikipedia-{WP_LANG}-train.txt.gz"
    OUT_TEST = f"wikipedia-{WP_LANG}-test.txt.gz"

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

print(f"Shuffle...")
dataset = dataset.shuffle(seed=42)

if NUM_SHARDS:
    print(f"Select shard...")
    dataset = dataset.shard(num_shards=NUM_SHARDS, index=SHARD_IDX)

print(f"Splitting...")
with gzip.open(OUT_TEST, "wt", encoding="utf-8") as f_test:
    with gzip.open(OUT_TRAIN, "wt", encoding="utf-8") as f_train:
        for i, row in enumerate(dataset):
            if i and not i % 100000:
                print(f"<processed {i} rows ({100*i/len(dataset):.2f}%) so far>")
            text = row['text'].lower().strip() + SEP
            if random.random() < TEST_SPLIT:
                f_test.write(text)
            else:
                f_train.write(text)

print(f"DONE.")
