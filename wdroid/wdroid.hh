#ifndef WDROID_HH
#define WDROID_HH

#include <map>
#include <array>
#include <vector>
#include <cstdint>
#include <cassert>
#include <cstdio>
#include <cstdlib>

extern const char WordleDroidWords4[];
extern const char WordleDroidWords5[];
extern const char WordleDroidWords6[];

template <int> static const char *getWordleDroidWords();
template <> inline const char *getWordleDroidWords<4>() { return WordleDroidWords4; }
template <> inline const char *getWordleDroidWords<5>() { return WordleDroidWords5; }
template <> inline const char *getWordleDroidWords<6>() { return WordleDroidWords6; }

struct AbstractWordleDroidEngine
{
};

template <int WordLen, int MaxCnt, int MaxWords>
struct WordleDroidEngine : public AbstractWordleDroidEngine
{
	static constexpr char Gray = 0;
	static constexpr char Yellow = 32;
	static constexpr char Green = 64;
	static constexpr char White = 96;

	struct Tok : public std::array<char, WordLen>
	{
		Tok() { }
		~Tok() { }

		Tok(const char *p) {
			for (int i=0; i<WordLen; i++)
				(*this)[i] = p[i];
		}

		char val(int idx) const { return (*this)[idx] & 31; }
		char col(int idx) const { return (*this)[idx] & 96; }

		void setCol(char newCol, int idx) {
			(*this)[idx] &= 31;
			(*this)[idx] |= newCol;
		}

		void setCol(char newCol) {
			for (int i=0; i<WordLen; i++)
				setCol(newCol, i);
		}

		bool operator<(const Tok& other) const {
			for (int i=0; i<WordLen; i++)
				if ((*this)[i] != other[i])
					return (*this)[i] < other[i];
			return false;
		}

		void print() const {
			for (int i=0; i<WordLen; i++)
				switch (col(i))
				{
				case Gray:
					printf("_%c", 64+val(i));
					break;
				case Yellow:
					printf(".%c", 64+val(i));
					break;
				case Green:
					printf("%c", 64+val(i));
					break;
				case White:
					printf("%c", 96+val(i));
					break;
				default:
					abort();
				}
		}
	};

	struct WordMsk : public std::array<int32_t, WordLen + MaxCnt>
	{
		WordMsk() { }
		~WordMsk() { }

		WordMsk(const Tok &w) {
			// w.print();
			// printf("\n");
			for (int k=0; k<MaxCnt; k++)
				cntBits(k) = 0;
			for (int i=0; i<WordLen; i++) {
				int32_t msk = 1 << w.val(i);
				posBits(i) = msk;
				// printf(" %d: 0x%x\n", i, msk);
				for (int k=0; k<MaxCnt; k++) {
					int32_t tmp = cntBits(k) & msk;
					// printf(" +: 0x%x 0x%x\n", cntBits(k), tmp);
					cntBits(k) |= msk;
					msk = tmp;
				}
				if (msk != 0 && msk != 1) {
					w.print();
					printf("\n");
					//assert(msk == 0 || msk == 1);
				}
			}
			int32_t msk = cntBits(MaxCnt-1);
			for (int k=MaxCnt-2; k >= 0; k--) {
				cntBits(k) &= ~msk;
				msk |= cntBits(k);
			}
		}
		int32_t &posBits(int idx) { return (*this)[idx]; }
		int32_t &cntBits(int idx) { return (*this)[WordLen+idx]; }
	};

	const WordMsk &word(const Tok &w) {
		auto it = wordIndex.find(w);
		if (it != wordIndex.end())
			return words[it->second];
		assert(numWords < MaxWords);
		words[numWords] = w;
		wordIndex[w] = numWords;
		return words[numWords++];
	}

	const WordMsk &dictWord(const Tok &w) {
		auto it = wordIndex.find(w);
		assert(it == wordIndex.end());
		assert(numWords < MaxWords);
		words[numWords] = w;
		wordIndex[w] = numWords;
		return words[numWords++];
	}

	void loadDict(const char *p) {
		while (*p) {
			dictWord(p);
			p += WordLen;
		}
	}

	void loadDefaultDict() {
		loadDict(getWordleDroidWords<WordLen>());
	}

	int numWords;
	std::vector<int> dictWords;
	std::map<Tok, int> wordIndex;
	std::array<WordMsk, MaxWords> words;

	WordleDroidEngine()
	{
		word(Tok()); // zero word
	}
};

extern template struct WordleDroidEngine<4, 3, 10000>;
using WordleDroidEngine4 = WordleDroidEngine<4, 3, 10000>;

extern template struct WordleDroidEngine<5, 3, 20000>;
using WordleDroidEngine5 = WordleDroidEngine<5, 3, 20000>;

extern template struct WordleDroidEngine<6, 4, 40000>;
using WordleDroidEngine6 = WordleDroidEngine<6, 4, 40000>;

#endif
