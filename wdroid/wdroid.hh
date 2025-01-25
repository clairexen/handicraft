#ifndef WDROID_HH
#define WDROID_HH

#include <array>
#include <vector>
#include <unordered_map>
#include <cstdint>

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
		Tok(const char *p) {
			for (int i=0; i<WordLen; i++)
				(*this)[i] = p[i];
		}

		char val(int idx) { return (*this)[idx] & 31; }
		char col(int idx) { return (*this)[idx] & 96; }

		void setCol(char newCol, int idx) {
			(*this)[idx] &= 31;
			(*this)[idx] |= newCol;
		}

		void setCol(char newCol) {
			for (int i=0; i<WordLen; i++)
				setCol(newCol, i);
		}
	};

	struct Word : public std::array<int32_t, WordLen + MaxCnt>
	{
		int32_t &posBits(int idx) { return (*this)[idx]; }
		int32_t &cntBits(int idx) { return (*this)[WordLen+idx]; }
	};

	int numWords;
	std::vector<int> dictWords;
	std::unordered_map<Tok, int> wordIndex;
	std::array<Word, MaxWords> words;
};
#endif
