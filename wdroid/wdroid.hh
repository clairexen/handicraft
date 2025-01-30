#ifndef WDROID_HH
#define WDROID_HH

#include <map>
#include <array>
#include <vector>
#include <string>
#include <cstdint>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <cstdlib>

extern const char WordleDroidWords4[];
extern const char WordleDroidWords5[];
extern const char WordleDroidWords6[];

template <int> static const char *getWordleDroidWords();
template <> inline const char *getWordleDroidWords<4>() { return WordleDroidWords4; }
template <> inline const char *getWordleDroidWords<5>() { return WordleDroidWords5; }
template <> inline const char *getWordleDroidWords<6>() { return WordleDroidWords6; }

struct WordleDroidGlobalState;

struct AbstractWordleDroidEngine
{
	WordleDroidGlobalState *globalState = nullptr;
	AbstractWordleDroidEngine(WordleDroidGlobalState *st) : globalState(st) { }
	virtual ~AbstractWordleDroidEngine() { }
	virtual int vGetWordLen() const { return 0; };
	virtual int vGetCurNumWords() const { return 0; };
	virtual bool vExecuteCommand(const char *p, const char *arg) { return false; };
};

struct WordleDroidGlobalState
{
	AbstractWordleDroidEngine *engine = nullptr;
	WordleDroidGlobalState() {
		engine = new AbstractWordleDroidEngine(this);
	}
	~WordleDroidGlobalState() {
		delete engine;
	}
	int main(int argc, const char **argv);
	bool executeCommand(const char *p, const char *arg, bool noprompt=false);
};

template <int WordLen, int MaxCnt, int MaxWords>
struct WordleDroidEngine : public AbstractWordleDroidEngine
{
	static constexpr char Gray = 0;
	static constexpr char Yellow = 32;
	static constexpr char Green = 64;
	static constexpr char White = 96;

	static constexpr int32_t FullMskVal = (1 << 27) - 2;

	struct Tok : public std::array<char, WordLen>
	{
		Tok() {
			for (int i=0; i<WordLen; i++)
				(*this)[i] = 0;
		}

		Tok(const char *p) {
			for (int i=0; i<WordLen; i++)
				(*this)[i] = p[i];
		}

		~Tok() { }

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

		static WordMsk fullMsk() {
			WordMsk w;
			for (int i=0; i<WordLen; i++)
				w.posBits(i) = FullMskVal;
			for (int k=0; k<MaxCnt; k++)
				w.cntBits(k) = FullMskVal;
			return w;
		}

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
					print();
					printf("\n");
					assert(msk == 0 || msk == 1);
				}
			}
			int32_t msk = cntBits(MaxCnt-1);
			for (int k=MaxCnt-2; k >= 0; k--) {
				cntBits(k) &= ~msk;
				msk |= cntBits(k);
			}
			// w.print();
			// printf(" ");
			// print();
			// printf("\n");
		}

		int32_t &posBits(int idx) { return (*this)[idx]; }
		int32_t &cntBits(int idx) { return (*this)[WordLen+idx]; }

		void printSingleMask(int32_t msk) {
			int popcnt = 0;
			for (int i = 1; i <= 26; i++)
				if (((msk >> i) & 1) != 0)
					popcnt++;
			if (popcnt <= 13) {
				for (int i = 1; i <= 26; i++)
					if (((msk >> i) & 1) != 0)
						printf("%c", 64 + i);
			} else {
				for (int i = 1; i <= 26; i++)
					if (((msk >> i) & 1) == 0)
						printf("%c", 96 + i);
			}
		}

		void print() {
			for (int i=0; i<WordLen; i++) {
				printf("/");
				printSingleMask(posBits(i));
			}
			for (int k=0; k<MaxCnt; k++) {
				printf(":");
				printSingleMask(cntBits(k));
			}
		}
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
		dictWords.push_back(numWords);
		words[numWords] = w;
		wordIndex[w] = numWords;
		return words[numWords++];
	}

	void loadDict(const char *p) {
		assert(dictWords.empty());
		while (*p) {
			dictWord(p);
			p += WordLen;
		}
		curWords = dictWords;
	}

	void loadDictFile(const char *p) {
		if (p == nullptr) {
			loadDict(getWordleDroidWords<WordLen>());
			return;
		}
		abort();
	}

	int numWords = 0;
	std::vector<int> dictWords;
	std::map<Tok, int> wordIndex;
	std::array<WordMsk, MaxWords> words;

	std::vector<int> curWords;
	WordMsk curWordMsk;

	WordleDroidEngine(WordleDroidGlobalState *st, const char *arg) : AbstractWordleDroidEngine(st)
	{
		word(Tok()); // zero word
		curWordMsk = WordMsk::fullMsk();
		loadDictFile(arg);
	}

	virtual int vGetWordLen() const final override { return WordLen; };

	virtual int vGetCurNumWords() const final override { return int(curWords.size()); };

	bool vExecuteCommand(const char *p, const char *arg) final override
	{
		using namespace std::string_literals;

		// if (p == "-D"s) { return true; }

		return false;
	}
};

extern template struct WordleDroidEngine<4, 3, 10000>;
using WordleDroidEngine4 = WordleDroidEngine<4, 3, 10000>;

extern template struct WordleDroidEngine<5, 3, 20000>;
using WordleDroidEngine5 = WordleDroidEngine<5, 3, 20000>;

extern template struct WordleDroidEngine<6, 4, 40000>;
using WordleDroidEngine6 = WordleDroidEngine<6, 4, 40000>;

bool WordleDroidGlobalState::executeCommand(const char *p, const char *arg, bool noprompt)
{
	using namespace std::string_literals;

	if (p == nullptr) {
		char buffer[1024];
		printf("[wdroid-%d] %5d> ", engine->vGetWordLen(), engine->vGetCurNumWords());
		fflush(stdout);
		if (fgets(buffer, 1024, stdin) == nullptr) {
			printf("-exit\n");
			delete engine;
			engine = nullptr;
			return true;
		}
		if (char *cursor = strchr(buffer, '\n'); cursor != nullptr)
			*cursor = 0;
		return executeCommand(buffer, nullptr, true);
	}

	if (arg == nullptr) {
		if (const char *s = strchr(p, '='); s != nullptr) {
			char *buffer = strdup(p);
			char *cursor = buffer + (s - p);
			*(cursor++) = 0;
			bool ret = executeCommand(buffer, cursor, noprompt);
			free(buffer);
			return ret;
		}
	}

	if (engine->vGetWordLen() == 0) {
		if (p != "-4"s && p != "-5"s && p != "-6"s && p != "-exit"s) {
			executeCommand("-5", nullptr);
			return executeCommand(p, arg);
		}
	}

	if (!noprompt)
		printf("[wdroid-%d] %5d> %s\n", engine->vGetWordLen(), engine->vGetCurNumWords(), p);

	if (p == "-4"s) {
		delete engine;
		engine = new WordleDroidEngine4(this, arg);
		return true;
	}

	if (p == "-5"s) {
		delete engine;
		engine = new WordleDroidEngine5(this, arg);
		return true;
	}

	if (p == "-6"s) {
		delete engine;
		engine = new WordleDroidEngine6(this, arg);
		return true;
	}

	if (p == "-exit"s) {
		delete engine;
		engine = nullptr;
		return true;
	}

	if (!engine->vExecuteCommand(p, arg)) {
		if (arg == nullptr)
			printf("Error executing command '%s'! Try -h for help.\n", p);
		else
			printf("Error executing command '%s' with arg '%s'! Try -h for help.\n", p, arg);
		return false;
	}

	return true;
}

int WordleDroidGlobalState::main(int argc, const char **argv)
{
	for (int i=1; engine != nullptr && i<argc; i++) {
		if (!executeCommand(argv[i], nullptr))
			return 1;
	}

	while (engine != nullptr)
		executeCommand(nullptr, nullptr);

	return 0;
}

#endif
