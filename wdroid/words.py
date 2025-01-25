import nltk
from nltk.corpus import wordnet, words

nltk.download('words')
all_en_words = set(words.words('en'))

nltk.download('wordnet')
all_wordnet_lemmas = set(wordnet.all_lemma_names())

dictfile = "/usr/share/dict/american-english-insane"
#dictfile = "/usr/share/dict/american-english-large"
with open(dictfile) as f:
    all_dict_words = {w.strip() for w in f}

all_words = all_en_words | all_wordnet_lemmas | all_dict_words

with open("words.cc", "w") as f:
    for N in (4, 5, 6):
        words = "".join(sorted([w for w in all_words if len(w) == N and
                                w.isalpha() and w.islower() and w.isascii()]))
        print(f"extern const char WordleDroidWords{N}[];", file=f)
        print(f"const char WordleDroidWords{N}[] = // {len(words)//N} words", end="", file=f)
        for k in range(0, len(words), 60):
            print(f"\n\"{words[k:k+60]}\"", end="", file=f)
        print(";", file=f)
