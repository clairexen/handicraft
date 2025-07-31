import datasets, tokens, numpy, sys, os
cfg = tokens.LogiChatConfig()
lex = tokens.gentokens(cfg)

dataset = datasets.load_dataset("openwebtext")
dataset = dataset["train"].train_test_split(test_size=0.001, seed=1357, shuffle=True)

if not os.access("datasets", F_OK):
    os.mkdir("datasets")

total = 0
rejected = 0
for name in ("test", "train"):
    print(f"Writing datasets/ds_openwebtext_{name}.b8 ...")
    with open(f"datasets/ds_openwebtext_{name}.b8", "w") as f:
        cnt = 0
        for d in dataset[name]:
            if cnt == 50000: break
            total += 1
            t = d["text"]
            if all(31 < ord(c) < 127 for c in t if c != "\n"):
                t = f"REM '''\n{t}\n'''\n"
                t = tokens.encode(lex, t) + [0]
                t = numpy.array(t, numpy.uint8)
                t.tofile(f)
                cnt += 1
            else:
                rejected += 1
        print(f"  written {cnt} items")

print(f"Rejected {rejected} / {total} items containing non-ASCII chars.")
