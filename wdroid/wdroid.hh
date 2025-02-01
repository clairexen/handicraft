// The New York Times "WordleBot" is behnd a paywall.  :/
// So I wrote my own "WordleDroid" which I can run locally.
//
// Copyright (C) 2025  Claire Xenia Wolf <claire@clairexen.net>
//
// Permission to use, copy, modify, and/or distribute this software for any
// purpose with or without fee is hereby granted, provided that the above
// copyright notice and this permission notice appear in all copies.
//
// THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
// WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
// ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
// WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
// ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
// OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

#ifndef WDROID_HH
#define WDROID_HH

#define ENABLE_WDROID_ENGINE_4
#define ENABLE_WDROID_ENGINE_5
#define ENABLE_WDROID_ENGINE_6

#include <map>
#include <array>
#include <ranges>
#include <vector>
#include <string>
#include <format>
#include <functional>
#include <iostream>
#include <fstream>
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
struct AbstractWordleDroidEngine;
template <int WordLen> struct WordleDroidEngine;


// =========================================================
// AbstractWordleDroidEngine with virtual interfaces and utility methods

struct AbstractWordleDroidEngine
{
	WordleDroidGlobalState *globalState = nullptr;

	AbstractWordleDroidEngine(WordleDroidGlobalState *st) : globalState(st) { }
	virtual ~AbstractWordleDroidEngine() { }
	virtual int vGetWordLen() const { return 0; };
	virtual int vGetCurNumWords() const { return 0; };
	virtual bool vExecuteCommand(const char *p, const char *arg,
			AbstractWordleDroidEngine* &nextEngine) { return false; };

	void pr(char c) const;
	void pr(const std::string &s) const;
	void prFlush() const;

	void prGrayTok()   const { pr("\033[30m\033[100m"); } // Black text, gray background
	void prYellowTok() const { pr("\033[30m\033[103m"); } // Black text, yellow background
	void prGreenTok()  const { pr("\033[37m\033[42m");  } // White text, green background
	void prWhiteTok()  const { pr("\033[30m\033[47m");  } // Black text, white background

	void prGrayFg()   const { pr("\033[90m"); } // Gray text
	void prYellowFg() const { pr("\033[93m"); } // Yellow text
	void prGreenFg()  const { pr("\033[32m"); } // Green text
	void prWhiteFg()  const { pr("\033[37m"); } // White text

	void prResetColors() const { pr("\033[0m"); }
	void prReplaceLastLine() const { pr("\033[F\033[2K"); }

	void prNl() const {
		pr('\n');
		prFlush();
	}

	int32_t grayKeyStatusBits = 0;
	int32_t yellowKeyStatusBits = 0;
	int32_t greenKeyStatusBits = 0;

	void prShowKeyboard() const {
		static const char *keysMap[3] = {
			"qwertyuiop",
			" asdfghjkl",
			"  zxcvbnm"
		};
		pr('\n');
		for (int i = 0; i < 3; i++) {
			for (const char *p = keysMap[i]; *p; p++) {
				if (*p == ' ') {
					pr("  ");
				} else {
					pr(' ');
					int32_t bitsMask = 1 << (*p & 31);
					if ((greenKeyStatusBits & bitsMask) != 0)
						prGreenFg();
					else if ((yellowKeyStatusBits & bitsMask) != 0)
						prYellowFg();
					else if ((grayKeyStatusBits & bitsMask) != 0)
						prGrayFg();
					else
						prWhiteFg();
					pr("▐");
					if ((greenKeyStatusBits & bitsMask) != 0)
						prGreenTok();
					else if ((yellowKeyStatusBits & bitsMask) != 0)
						prYellowTok();
					else if ((grayKeyStatusBits & bitsMask) != 0)
						prGrayTok();
					else
						prWhiteTok();
					pr(*p - 32);
					if ((greenKeyStatusBits & bitsMask) != 0)
						prGreenFg();
					else if ((yellowKeyStatusBits & bitsMask) != 0)
						prYellowFg();
					else if ((grayKeyStatusBits & bitsMask) != 0)
						prGrayFg();
					else
						prWhiteFg();
					pr("▌");
					prResetColors();
				}
			}
			pr('\n');
		}
		pr('\n');
	}

	void prHideKeyboard() const {
		// pr("\n\033[2K");
		// pr("\n\033[2K");
		// pr("\033[F\033[F");
	}

	void prPrompt() const {
		pr(std::format("[wdroid-{}] {:5}> ", vGetWordLen(), vGetCurNumWords()));
		prFlush();
	}

	int scanWord(const char *p) const {
		int n = 0;
		while ('a' <= *p && *p <= 'z') n++, p++;
		return n;
	}

	int scanTag(const char *p, const char *q) const {
		int n = 0;
		while (('a' <= *q && *q <= 'z') && (('0' <= *p && *p <= '2') || (*p == '.') ||
				(*p == '_') || ('A' <= *p && *p <= 'Z') || (*p == *q)))
			n++, p++, q++;
		return n;
	}

	bool boolArg(const char *arg) {
		using namespace std::string_literals;
		if (arg == nullptr) return true;
		if (*arg == 0) return false;
		if (*arg == '0') return false;
		if (arg == "off"s) return false;
		return true;
	}

	int intArg(const char *arg, int onVal=1, int offVal=0) {
		if (arg == nullptr) return onVal;
		if (*arg == 0) return offVal;
		return atoi(arg);
	}
};

struct WordleDroidGlobalState
{
	AbstractWordleDroidEngine *engine = nullptr;
	std::ofstream outfile;
	bool refineMasks = false;
	bool showMasks = false;
	int showLists = 0;
	int showKeys = 0;

	WordleDroidGlobalState() {
		engine = new AbstractWordleDroidEngine(this);
	}

	~WordleDroidGlobalState() {
		delete engine;
	}

	int main(int argc, const char **argv);
	bool executeCommand(const char *p, const char *arg, bool noprompt=false);
};


// =========================================================
// Main WordleDroidEngine template class

template <int WordLen>
struct WordleDroidEngine : public AbstractWordleDroidEngine
{
	static constexpr char MaxCnt =
			WordLen == 4 ? 4 :
			WordLen == 5 ? 4 :
			WordLen == 6 ? 5 : -1;

	static constexpr char Gray = 0;
	static constexpr char Yellow = 32;
	static constexpr char Green = 64;
	static constexpr char White = 96;

	static constexpr int32_t FullMskVal = (1 << 27) - 2;


	// =========================================================
	// Word/Guess/Hint Token Data Structure

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

		Tok(const char *p, const char *q) {
			std::vector<int> freeChars;
			for (int i=0; i<WordLen; i++) {
				(*this)[i] = p[i] & 31;
				if (q[i] == '1') (*this)[i] += 32;
				if (q[i] == '2') (*this)[i] += 64;
				if ('a' <= q[i] && q[i] <= 'z') (*this)[i] += 32;
				if ('A' <= q[i] && q[i] <= 'Z') {
					if (((p[i] ^ q[i]) & 31) == 0)
						(*this)[i] += 64;
					else
						freeChars.push_back(i);
				}
			}
			for (int idx : freeChars) {
				char ch = q[idx] & 31;
				for (int i=0; i<WordLen; i++)
					if ((*this)[i] == ch) {
						(*this)[i] += 32;
						break;
					}
			}
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

	void prTok(const Tok &tok)
	{
		for (int i=0; i<WordLen; i++)
			switch (tok.col(i))
			{
			case Gray:
				prGrayTok();
				pr(64+tok.val(i));
				break;
			case Yellow:
				prYellowTok();
				pr(64+tok.val(i));
				break;
			case Green:
				prGreenTok();
				pr(64+tok.val(i));
				break;
			case White:
				prWhiteTok();
				pr(96+tok.val(i));
				break;
			default:
				abort();
			}
		prResetColors();
	}

	void applyHintToKeyStatusBits(const Tok &hint)
	{
		for (int i=0; i < WordLen; i++)
		{
			switch (hint.col(i))
			{
			case Gray:
				grayKeyStatusBits |= 1 << hint.val(i);
				break;
			case Yellow:
				yellowKeyStatusBits |= 1 << hint.val(i);
				break;
			case Green:
				greenKeyStatusBits |= 1 << hint.val(i);
				break;
			}
		}
	}


	// =========================================================
	// Word BitMasks Data Structure

	struct WordMsk : public std::array<int32_t, WordLen + MaxCnt>
	{
		WordMsk() { }
		~WordMsk() { }

		static constexpr WordMsk fullMsk() {
			WordMsk w;
			for (int i=0; i<WordLen; i++)
				w.posBits(i) = FullMskVal;
			for (int k=0; k<MaxCnt; k++)
				w.cntBits(k) = FullMskVal;
			return w;
		}

		WordMsk(const Tok &w)
		{
			for (int i=0; i<WordLen; i++)
				posBits(i) = FullMskVal;
			for (int k=0; k<MaxCnt; k++)
				cntBits(k) = k ? 0 : FullMskVal;

			int numgray = 0;
			int32_t graymsk = 0;
			for (int i=0; i<WordLen; i++)
			{
				int32_t msk = (1 << w.val(i)) & FullMskVal;

				switch (w.col(i))
				{
				case Green:
					posBits(i) = msk;
					if (0)
				case Yellow:
						posBits(i) &= ~msk;
					assert((cntBits(MaxCnt-1) & msk) == 0);
					for (int k=MaxCnt-1; k > 0; k--) {
						cntBits(k) |= cntBits(k-1) & msk;
						cntBits(k-1) &= ~msk;
					}
					break;
				case Gray:
					graymsk |= msk;
					numgray++;
					break;
				case White:
				default:
					abort();
				}
			}

			// grow cntbits upwards as appropiate
			for (int i = 0; i < numgray; i++) {
				for (int k=MaxCnt-1; k > 0; k--) {
					cntBits(k) |= cntBits(k-1) & ~graymsk;
				}
			}
		}

		int32_t posBits(int idx) const { return (*this)[idx]; }
		int32_t cntBits(int idx) const { return (*this)[WordLen+idx]; }

		int32_t &posBits(int idx) { return (*this)[idx]; }
		int32_t &cntBits(int idx) { return (*this)[WordLen+idx]; }

		void add(const WordMsk &other) {
			for (int i=0; i<WordLen; i++)
				posBits(i) |= other.posBits(i);
			for (int k=0; k<MaxCnt; k++)
				cntBits(k) |= other.cntBits(k);
		}

		void intersect(const WordMsk &other) {
			for (int i=0; i<WordLen; i++)
				posBits(i) &= other.posBits(i);
			for (int k=0; k<MaxCnt; k++)
				cntBits(k) &= other.cntBits(k);
		}

		bool match(const WordMsk &other) const {
			int32_t failBits = 0;
			for (int i=0; i<WordLen; i++)
				failBits |= (posBits(i) & other.posBits(i)) ^ other.posBits(i);
			for (int k=0; k<MaxCnt; k++)
				failBits |= (cntBits(k) & other.cntBits(k)) ^ other.cntBits(k);
			return failBits == 0;
		}
	};

	void prSingleMask(int32_t msk) const {
		int popcnt = 0;
		for (int i = 1; i <= 26; i++)
			if (((msk >> i) & 1) != 0)
				popcnt++;
		if (popcnt == 26) {
			pr('*');
			return;
		}
		if (popcnt <= 13) {
			for (int i = 1; i <= 26; i++)
				if (((msk >> i) & 1) != 0)
					pr(64 + i);
			return;
		}
		for (int i = 1; i <= 26; i++)
			if (((msk >> i) & 1) == 0)
				pr(96 + i);
	}

	void prWordMsk(const WordMsk &w) const {
		for (int i=0; i<WordLen; i++) {
			pr('/');
			prSingleMask(w.posBits(i));
		}
		for (int k=0; k<MaxCnt; k++) {
			pr(':');
			prSingleMask(w.cntBits(k));
		}
	}


	// =========================================================
	// Main State (wordsMsk and wordsList) and related helper functions

	struct WordData {
		int idx;
		Tok tok;
		WordMsk msk;
	};

	WordMsk wordsMsk;
	std::vector<int> allWords;
	std::map<Tok, int> wordsIndex;
	std::vector<WordData> wordsList;

	auto words() const { return wordsList | std::views::drop(1); }

	int findWord(Tok w) const {
		w.setCol(Green);
		auto it = wordsIndex.find(w);
		if (it == wordsIndex.end())
			return 0;
		return it->second;
	}

	std::pair<std::vector<int>, WordMsk>
	filterWords(const std::vector<int> &oldWords, const WordMsk &msk)
	{
		std::pair<std::vector<int>, WordMsk> ret;
		auto& [newWords, newMsk] = ret;

		newWords.reserve(oldWords.size());
		for (int idx : oldWords) {
			if (msk.match(wordsList[idx].msk)) {
				newWords.push_back(idx);
				newMsk.add(wordsList[idx].msk);
			}
		}
		return ret;
	}


	// =========================================================
	// Constructors and related helper functions

	bool addWord(Tok w) {
		w.setCol(Green);
		auto it = wordsIndex.find(w);
		if (it != wordsIndex.end())
			return false;
		int idx = wordsList.size();
		wordsList.emplace_back(idx, w, w);
		if (idx == 0)
			return true;
		allWords.push_back(idx);
		wordsIndex[w] = idx;
		return true;
	}

	void loadDict(const char *p) {
		assert(wordsList.size() == 1);
		while (*p) {
			addWord(p);
			p += WordLen;
		}
	}

	void loadDictFile(const char *p) {
		if (p == nullptr) {
			loadDict(getWordleDroidWords<WordLen>());
			return;
		}
		abort();
	}

	WordleDroidEngine(WordleDroidGlobalState *st, const char *arg) : AbstractWordleDroidEngine(st)
	{
		addWord(Tok()); // zero word
		wordsMsk = WordMsk::fullMsk();
		loadDictFile(arg);
	}

	WordleDroidEngine(WordleDroidEngine *parent, WordMsk msk = WordMsk::fullMsk()) :
		AbstractWordleDroidEngine(parent->globalState)
	{
		wordsList.reserve(parent->wordsList.size());
		allWords.reserve(parent->wordsList.size());
		addWord(Tok()); // zero word

		WordMsk refinedMsk;
		msk.intersect(parent->wordsMsk);

		for (auto &wdata : parent->words()) {
			if (!msk.match(wdata.msk))
				continue;
			int idx = wordsList.size();
			wordsList.emplace_back(idx, wdata.tok, wdata.msk);
			wordsIndex[wdata.tok] = idx;
			allWords.push_back(idx);
			refinedMsk.add(wdata.msk);
		}

		wordsMsk = globalState->refineMasks ? refinedMsk : msk;

		grayKeyStatusBits = parent->grayKeyStatusBits;
		yellowKeyStatusBits = parent->yellowKeyStatusBits;
		greenKeyStatusBits = parent->greenKeyStatusBits;
	}


	// =========================================================
	// Command and Extension Registry

	typedef std::function<AbstractWordleDroidEngine*(WordleDroidEngine*)> cmdLoad_t;

	static std::map<std::string, cmdLoad_t>& cmdTable() {
		static std::map<std::string, cmdLoad_t> staticData;
		return staticData;
	}

	static int regCmd(std::initializer_list<std::string> names, cmdLoad_t func) {
		auto &db = cmdTable();
		for (auto &n : names)
			db[n] = func;
		return db.size();
	}


	// =========================================================
	// Command executer and other virtual methods

	virtual int vGetWordLen() const final override { return WordLen; };

	virtual int vGetCurNumWords() const final override { return int(wordsList.size()-1); };

	bool tryExecuteHintCommand(const char *p, const char *arg,
			AbstractWordleDroidEngine* &nextEngine)
	{
		const char *p1 = p;
		const char *p2 = p1 + WordLen;
		const char *p3 = p2 + 1;
		const char *p4 = p3 + WordLen;

		if (scanWord(p1) != WordLen) return false;
		if (*p2 != '/') return false;
		if (scanTag(p3, p1) != WordLen) return false;
		if (*p4 != 0) return false;

		Tok hint(p, p+WordLen+1);
		applyHintToKeyStatusBits(hint);

		prReplaceLastLine();
		prPrompt();
		prTok(hint);

		auto ne = new WordleDroidEngine(this, hint);
		nextEngine = ne;

		if (globalState->showMasks)
			pr(' '), prWordMsk(ne->wordsMsk);

		prNl();

		return true;
	}

	bool vExecuteCommand(const char *p, const char *arg,
			AbstractWordleDroidEngine *&nextEngine) override
	{
		using namespace std::string_literals;

		if (p == "-l"s) {
			int cnt = 0;
			for (auto &wdata : words()) {
				pr(cnt % 16 ? " " : cnt ? "\n  " : "  ");
				Tok tok = wdata.tok;
				tok.setCol(White);
				prTok(tok);
				cnt++;
			}
			prNl();
			return true;
		}

		if (tryExecuteHintCommand(p, arg, nextEngine))
			return true;

		return false;
	}
};


// =========================================================
// Explicit engine declarations (see wdroid.cc for instantiations)

#define REG_WDROID_N_CMDS(extType_, wordLen_, ...) \
static int REG_WDROID_CMDS_ ## extType_ ## _ ## wordLen_ = \
WordleDroidEngine ## wordLen_::regCmd({__VA_ARGS__}, \
[](WordleDroidEngine ## wordLen_* parent_) -> AbstractWordleDroidEngine* \
{ return new extType_<wordLen_>(parent_); });

#ifdef ENABLE_WDROID_ENGINE_4
extern template struct WordleDroidEngine<4>;
using WordleDroidEngine4 = WordleDroidEngine<4>;
#define REG_WDROID4_CMDS(extType_, ...) \
  REG_WDROID_N_CMDS(extType_, 4, __VA_ARGS__)
#else
#define REG_WDROID4_CMDS(extType_, ...)
#endif

#ifdef ENABLE_WDROID_ENGINE_5
extern template struct WordleDroidEngine<5>;
using WordleDroidEngine5 = WordleDroidEngine<5>;
#define REG_WDROID5_CMDS(extType_, ...) \
  REG_WDROID_N_CMDS(extType_, 5, __VA_ARGS__)
#else
#define REG_WDROID5_CMDS(extType_, ...)
#endif

#ifdef ENABLE_WDROID_ENGINE_6
extern template struct WordleDroidEngine<6>;
using WordleDroidEngine6 = WordleDroidEngine<6>;
#define REG_WDROID6_CMDS(extType_, ...) \
  REG_WDROID_N_CMDS(extType_, 6, __VA_ARGS__)
#else
#define REG_WDROID6_CMDS(extType_, ...)
#endif

#define REG_WDROID_CMDS(extType_, ...) \
  REG_WDROID4_CMDS(extType_, __VA_ARGS__) \
  REG_WDROID5_CMDS(extType_, __VA_ARGS__) \
  REG_WDROID6_CMDS(extType_, __VA_ARGS__)

#endif
