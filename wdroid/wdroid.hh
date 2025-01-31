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

#include <map>
#include <array>
#include <vector>
#include <string>
#include <format>
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

struct AbstractWordleDroidEngine
{
	WordleDroidGlobalState *globalState = nullptr;
	AbstractWordleDroidEngine(WordleDroidGlobalState *st) : globalState(st) { }
	virtual ~AbstractWordleDroidEngine() { }
	virtual int vGetWordLen() const { return 0; };
	virtual int vGetCurNumWords() const { return 0; };
	virtual bool vExecuteCommand(const char *p, const char *arg) { return false; };

	void pr(char c) const;
	void pr(const std::string &s) const;
	void prFlush() const;

	void prReplaceLastLine() const { pr("\033[F\033[2K"); }
	void prResetColors() const { pr("\033[0m"); }

	void prGrayTok()   const { pr("\033[30m\033[100m"); } // Black text, gray background
	void prYellowTok() const { pr("\033[30m\033[43m");  } // Black text, yellow background
	void prGreenTok()  const { pr("\033[37m\033[42m");  } // White text, green background
	void prWhiteTok()  const { pr("\033[30m\033[47m");  } // Black text, white background

	void prGrayFg()   const { pr("\033[90m"); } // Gray text
	void prYellowFg() const { pr("\033[33m"); } // Yellow text
	void prGreenFg()  const { pr("\033[32m"); } // Green text
	void prWhiteFg()  const { pr("\033[37m"); } // White text

	void prNl() const {
		pr('\n');
		prFlush();
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
};

struct WordleDroidGlobalState
{
	AbstractWordleDroidEngine *engine = nullptr;
	std::ofstream outfile;

	WordleDroidGlobalState() {
		engine = new AbstractWordleDroidEngine(this);
	}

	~WordleDroidGlobalState() {
		delete engine;
	}

	int main(int argc, const char **argv);
	bool executeCommand(const char *p, const char *arg, bool noprompt=false);
};

void AbstractWordleDroidEngine::pr(char c) const {
	if (globalState && globalState->outfile.is_open())
		globalState->outfile << c;
	else
		std::cout << c;
}

void AbstractWordleDroidEngine::pr(const std::string &s) const {
	if (globalState && globalState->outfile.is_open())
		globalState->outfile << s;
	else
		std::cout << s;
}

void AbstractWordleDroidEngine::prFlush() const {
	if (globalState && globalState->outfile.is_open())
		globalState->outfile << std::flush;
	else
		std::cout << std::flush;
}

template <int WordLen, int MaxCnt>
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

	struct DictWordData {
		int idx;
		Tok tok;
		WordMsk msk;
	};

	std::vector<DictWordData> dictWords;
	std::map<Tok, int> dictWordIndex;

	std::vector<int> curWords;
	WordMsk curWordMsk;

	int findDictWord(Tok w) const {
		w.setCol(Green);
		auto it = dictWordIndex.find(w);
		if (it == dictWordIndex.end())
			return 0;
		return it->second;
	}

	bool addDictWord(Tok w) {
		w.setCol(Green);
		auto it = dictWordIndex.find(w);
		if (it != dictWordIndex.end())
			return false;
		int idx = dictWords.size();
		dictWords.emplace_back(idx, w, w);
		if (idx == 0)
			return true;
		curWords.push_back(idx);
		dictWordIndex[w] = idx;
		return true;
	}

	void loadDict(const char *p) {
		assert(dictWords.size() == 1);
		while (*p) {
			addDictWord(p);
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

	std::vector<int> filterWords(const std::vector<int> oldWords, const WordMsk &msk)
	{
		std::vector<int> newWords;
		newWords.reserve(oldWords.size());
		for (int idx : oldWords) {
			if (msk.match(dictWords[idx].msk))
				newWords.push_back(idx);
		}
		return newWords;
	}

	WordleDroidEngine(WordleDroidGlobalState *st, const char *arg) : AbstractWordleDroidEngine(st)
	{
		addDictWord(Tok()); // zero word
		loadDictFile(arg);
		curWordMsk = WordMsk::fullMsk();
	}

	virtual int vGetWordLen() const final override { return WordLen; };

	virtual int vGetCurNumWords() const final override { return int(curWords.size()); };

	bool tryExecuteHintCommand(const char *p, const char *arg)
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
		prReplaceLastLine();
		prPrompt();
		prTok(hint);
		pr(' ');

		WordMsk msk(hint);
		// prWordMsk(msk);
		// pr(' ');

		curWordMsk.intersect(msk);
		prWordMsk(curWordMsk);
		prNl();

		curWords = filterWords(curWords, curWordMsk);

		return true;
	}

	bool vExecuteCommand(const char *p, const char *arg) final override
	{
		using namespace std::string_literals;

		if (p == "-l"s) {
			int cnt = 0;
			for (auto idx : curWords) {
				pr(cnt % 16 ? " " : cnt ? "\n  " : "  ");
				Tok tok = dictWords[idx].tok;
				tok.setCol(White);
				prTok(tok);
				cnt++;
			}
			prNl();
			return true;
		}

		if (tryExecuteHintCommand(p, arg))
			return true;

		return false;
	}
};

extern template struct WordleDroidEngine<4, 4>;
using WordleDroidEngine4 = WordleDroidEngine<4, 4>;

extern template struct WordleDroidEngine<5, 4>;
using WordleDroidEngine5 = WordleDroidEngine<5, 4>;

extern template struct WordleDroidEngine<6, 5>;
using WordleDroidEngine6 = WordleDroidEngine<6, 5>;

bool WordleDroidGlobalState::executeCommand(const char *p, const char *arg, bool noprompt)
{
	using namespace std::string_literals;

	if (p == nullptr) {
		char buffer[1024];
		engine->prPrompt();
		if (fgets(buffer, 1024, stdin) == nullptr) {
			engine->pr("-exit\n");
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

	if (!noprompt) {
		engine->prPrompt();
		engine->pr(p);
		engine->prNl();
	}

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
