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

#include "wdroid.hh"

template <int WordLen>
struct WordleDroidMinMax : public WordleDroidEngine<WordLen>
{
	using Base = WordleDroidEngine<WordLen>;

	using typename Base::Tok;
	using typename Base::WordMsk;
	using typename Base::WordMskHash;

	using Base::pr;
	using Base::prNl;
	using Base::prTok;
	using Base::prFlush;
	using Base::refinedWordsMsk;
	using Base::White;
	using Base::wordsList;
	using Base::findWord;
	using Base::refineWords;
	using Base::refineMask;
	using Base::intArg;
	using Base::rng;

	struct StateData
	{
		int idx;
		WordMsk msk;
		std::vector<int> words;

		// filled in by expandState():
		// children[guessWordIdx] = { unique_sorted_next_states }
		std::vector<std::vector<int>> children;

		int pathDepth = 1 << 30;
		int pathParent = 0;
		int pathGuess = 0;

		// filled in by min-max sweep (and expandState() for traps)
		// (set to depth>=0 when state is queued or marked terminal)
		int depth = -1;

		int perfectDepth = 0;
		std::vector<int> perfectParents;
		std::vector<int> perfectChildren;
	};

	bool verbose = true;

	void vPr(char c) const { if (verbose) pr(c); }
	void vPr(const std::string &s) const { if (verbose) pr(s); }
	void vPrNl() const { if (verbose) prNl(); }

	bool setupDone = false;
	int maxNumStates = 100000000;
	int minStateSize = 0;
	int maxSearchDepth = 0;
	int maxTraceLength = 0;
	int firstGuessIdx = 0;
	std::vector<int> depthSizeLimits;

	std::vector<int> trapStates;
	std::vector<int> terminalStates;
	std::vector<int> nonTerminalStates;
	std::vector<StateData> stateList;
	std::unordered_map<WordMsk, int, WordMskHash> stateIndex;
	std::priority_queue<std::pair<int, int>> stateQueue;
	std::vector<std::vector<WordMsk>> hintMskTab;

	int largestSimpleTrapState[WordLen] = { };
	int largestComplexTrapState[WordLen] = { };

	int getStateIdx(const std::vector<int> &srcWords, const WordMsk &srcMsk)
	{
		if (auto it = stateIndex.find(srcMsk); it != stateIndex.end())
			return it->second;

		auto [wl, msk] = refineWords(srcWords, srcMsk);

		if (auto it = stateIndex.find(msk); it != stateIndex.end()) {
			int idx = it->second;
			stateIndex[srcMsk] = idx;
			return idx;
		}

		int idx = stateList.size();
		stateIndex[srcMsk] = idx;
		stateIndex[msk] = idx;
		stateList.emplace_back(idx, msk, wl);
		auto *state = &stateList[idx];

		// detecting "simple traps"
		if (int nwords = state->words.size(); idx != 0 && nwords <= 26) {
			int seenBits = 0, lockedCnt = 0;
			for (int i = 0; i < WordLen; i++) {
				int bits = state->msk.posBits(i);
				int nbits = std::popcount(uint32_t(bits));
				if (nbits == 1) {
					lockedCnt++;
					continue;
				}
				if (nbits != nwords || (seenBits & bits) != 0)
					goto not_a_simple_trap;
				seenBits |= bits;
			}
			state->depth = nwords;
			trapStates.push_back(idx);
			terminalStates.push_back(idx);
			if (lockedCnt < WordLen && (!largestSimpleTrapState[lockedCnt] || nwords >
					stateList[largestSimpleTrapState[lockedCnt]].depth))
				largestSimpleTrapState[lockedCnt] = idx;
			return idx;
		not_a_simple_trap:;
		}

		if (wl.size() < std::max(minStateSize, 2)) {
			state->depth = 0;
			terminalStates.push_back(idx);
			return idx;
		}

		if (idx == 0) {
			state->depth = 0;
			state->pathDepth = 0;
			stateQueue.emplace(wl.size(), idx);
		}
		return idx;
	}

	void expandState(int idx, int guessWordIdx = 0)
	{
		auto *state = &stateList[idx];
		std::vector<int> trapGuesses;

		if (maxSearchDepth > 0 && state->pathDepth > maxSearchDepth) {
			terminalStates.push_back(idx);
			return;
		}

		int removedLimitedGuesses = 0;
		int limit = state->words.size();
		if (state->pathDepth < depthSizeLimits.size())
			limit = depthSizeLimits[state->pathDepth];

		state->children.resize(state->words.size());
		for (int i = 0; i < state->words.size(); i++)
		{
			int ki = state->words[i];
			if (guessWordIdx && ki != guessWordIdx)
				continue;
			if (!state->children[i].empty())
				continue;

			std::vector<int> vec;
			vec.reserve(state->words.size());
			int maxChildSize = 0;
			for (int j = 0; j < state->words.size(); j++) {
				int kj = state->words[j];
				if (ki == kj && state->words.size() > 1)
					continue;
				WordMsk msk = state->msk;
				msk.intersect(hintMskTab[ki][kj]);
				msk.cleanup();
				int childIdx = getStateIdx(state->words, msk);
				state = &stateList[idx]; // invalidated by getStateIdx()
				auto &st = stateList[childIdx];
				if (st.pathDepth > state->pathDepth+1) {
					st.pathDepth = state->pathDepth+1;
					st.pathParent = idx;
					st.pathGuess = i;
				}
				if (maxTraceLength > 0 && st.depth > 0) {
					int traceLen = state->pathDepth + st.depth + 1;
					if (traceLen >= maxTraceLength) {
						j = state->words.size();
						maxChildSize = 0;
						vec.clear();
					}
				}
				maxChildSize = std::max(maxChildSize, int(st.words.size()));
				vec.push_back(childIdx);
			}

			if (maxChildSize > limit) {
				removedLimitedGuesses++;
				continue;
			}

			std::ranges::sort(vec);
			auto subrange = std::ranges::unique(vec);
			vec.erase(subrange.begin(), subrange.end());
			vec.shrink_to_fit();
			if (vec.size() == 1 && state->words.size()-1 ==
					stateList[vec.front()].words.size())
				trapGuesses.push_back(i);
			state->children[i] = std::move(vec);
		}

		if (removedLimitedGuesses == state->words.size()) {
			state->children.clear();
			terminalStates.push_back(idx);
			return;
		}

		// detecting "complex traps"
		if (trapGuesses.size() == state->words.size())
		{
			state->depth = state->words.size();
			state->children.clear();
			trapStates.push_back(idx);
			terminalStates.push_back(idx);

			int lockedCnt = 0;
			for (int i = 0; i < WordLen; i++) {
				int bits = state->msk.posBits(i);
				int nbits = std::popcount(uint32_t(bits));
				if (nbits == 1)
					lockedCnt++;
			}

			if (lockedCnt < WordLen && (!largestComplexTrapState[lockedCnt] ||
					state->depth > stateList[largestComplexTrapState[lockedCnt]].depth))
				largestComplexTrapState[lockedCnt] = idx;
			return;
		}

		if (trapGuesses.size() + removedLimitedGuesses < state->children.size())
			for (int i : trapGuesses)
				state->children[i].clear();

		for (int i = 0; i < state->words.size(); i++)
			for (int k : state->children[i]) {
				auto &st = stateList[k];
				if (st.depth < 0) {
					st.depth = 0;
					stateQueue.emplace(st.words.size(), k);
				}
			}

		nonTerminalStates.push_back(idx);
	}

	WordleDroidMinMax(Base *parent) : Base(parent)
	{
	}

	void doPrintBestGuessesByChildSize(int idx, int maxNum)
	{
		auto &state = stateList[idx];
		std::vector<std::pair<int,int>> guesses;
		for (int i = 0; i < state.children.size(); i++) {
			if (state.children[i].empty())
				continue;
			int maxSize = 0;
			for (int k : state.children[i])
				maxSize = std::max(maxSize, int(stateList[k].words.size()));
			guesses.emplace_back(maxSize, state.words[i]);
		}
		std::sort(guesses.begin(), guesses.end());
		for (int i = 0; i < guesses.size() && i < maxNum; i++) {
			if (i != 0 && i % 5 == 0)
				prNl();
			pr(std::format("{:6} ", guesses[i].first));
			Tok t = wordsList[guesses[i].second].tok;
			t.setCol(White);
			prTok(t);
		}
		prNl();
	}

	void doSetup()
	{
		if (setupDone)
			return;

		vPr("Creating hint-mask table...\n");
		hintMskTab.resize(wordsList.size());
		std::unordered_map<WordMsk, WordMsk, WordMskHash> refineCache;
		std::unordered_set<WordMsk, WordMskHash> uniqueRefinedMasks;
		for (int i = 1; i < wordsList.size(); i++) {
			const char *p = wordsList[i].tok.begin();
			hintMskTab[i].resize(wordsList.size());
			for (int j = 1; j < wordsList.size(); j++) {
				const char *q = wordsList[j].tok.begin();
				WordMsk msk = refinedWordsMsk;
				msk.intersect(Tok(p, q));
				if (auto it = refineCache.find(msk); it != refineCache.end())
					msk = it->second;
				else
					msk = refineCache[msk] = refineMask(msk);
				uniqueRefinedMasks.insert(msk);
				hintMskTab[i][j] = msk;
			}
		}
		vPr(std::format("  unique refined masks:    {:10}\n", uniqueRefinedMasks.size()));
		vPr(std::format("  total number of masks:   {:10}\n", refineCache.size()));
		vPr(std::format("  total size of table:   {:12}\n", wordsList.size()*wordsList.size()));

		vPr("Creating root state...\n");
		std::vector<int> rootWordsList;
		rootWordsList.reserve(wordsList.size());
		for (int i = 1; i < wordsList.size(); i++)
			rootWordsList.push_back(i);
		assert(getStateIdx(rootWordsList, WordMsk::fullMsk()) == 0);
		if (!stateQueue.empty())
			vPr(std::format("  size of largest queued state: {:5}\n", stateQueue.top().first));
		vPr(std::format("  total number of (new) states: {:5}\n", stateList.size()));

		vPr("Creating all first-level child states...\n");
		int prevNumStates = stateList.size();
		assert(stateQueue.top().second == 0);
		stateQueue.pop();
		expandState(0, firstGuessIdx);

		if (!stateQueue.empty())
			vPr(std::format("  size of largest queued state: {:5}\n", stateQueue.top().first));
		std::string tmpStr = std::format("{}/{}", stateList.size()-prevNumStates, 1);
		vPr(std::format("  new/processed states: {:>13}\n", tmpStr));
		vPr(std::format("  number of queued states:  {:9}\n", stateQueue.size()));
		vPr(std::format("  number of terminal states: {:8}\n", terminalStates.size()));
		vPr(std::format("  total number of states:  {:10}\n", stateList.size()));
		vPr(std::format("  total number of masks:  {:11}\n", stateIndex.size()));

		setupDone = true;
	}

	bool doBatch()
	{
		int batchSize = 100000;
		if (!stateQueue.empty()) {
			if (stateQueue.top().first >    5) batchSize = 50000;
			if (stateQueue.top().first >    7) batchSize = 20000;
			if (stateQueue.top().first >   10) batchSize = 10000;
			if (stateQueue.top().first >   25) batchSize = 5000;
			if (stateQueue.top().first >   50) batchSize = 2000;
			if (stateQueue.top().first >   75) batchSize = 1000;
			if (stateQueue.top().first >  100) batchSize = 500;
			if (stateQueue.top().first >  250) batchSize = 200;
			if (stateQueue.top().first >  500) batchSize = 100;
			if (stateQueue.top().first >  750) batchSize = 50;
			if (stateQueue.top().first > 1000) batchSize = 20;
			if (stateQueue.top().first > 2000) batchSize = 10;
			if (stateQueue.top().first > 3000) batchSize = 5;
			if (stateQueue.top().first > 4000) batchSize = 2;
		}

		vPr(std::format("Processing next batch of up to {} states...\n", batchSize));
		int i, prevNumStates = stateList.size();
		for (i=0; i<batchSize; i++) {
			if (stateQueue.empty()) break;
			if (stateList.size() >= maxNumStates) break;
			int idx = stateQueue.top().second;
			stateQueue.pop();
			expandState(idx);
		}

		if (!stateQueue.empty())
			vPr(std::format("  size of largest queued state: {:5}\n", stateQueue.top().first));
		std::string tmpStr = std::format("{}/{}", stateList.size()-prevNumStates, i);
		vPr(std::format("  new/processed states: {:>13}\n", tmpStr));
		vPr(std::format("  number of queued states:  {:9}\n", stateQueue.size()));
		vPr(std::format("  number of terminal states: {:8}\n", terminalStates.size()));
		vPr(std::format("  total number of states:  {:10}\n", stateList.size()));
		vPr(std::format("  total number of masks:  {:11}\n", stateIndex.size()));

		return i == batchSize;
	}

	std::string getPath(int idx)
	{
		std::string s;

		const auto &secret = wordsList[stateList[idx].words.front()];

		while (idx)
		{
			const auto &state = stateList[idx];
			const auto &guess = wordsList[stateList[state.pathParent].words[state.pathGuess]];
			std::string tmp;
			for (int i = 0; i < WordLen; i++)
				tmp += char(96 + guess.tok.val(i));
			if (s.empty()) {
				s = tmp + '/';
				Tok hint(guess.tok.data(), secret.tok.data());
				for (int i = 0; i < WordLen; i++)
					s += char('0' + hint.col(i)/32);
			} else {
				s = tmp + '/' + s;
			}
			idx = state.pathParent;
		}
		return s;
	}


	void doShowTraps()
	{
		pr("Largest found simple/complex trap states by number of variable chars:\n");

		std::vector<std::vector<int>> data;
		std::vector<int> dataStates;
		std::vector<Tok> secrets;

		for (int i=WordLen-1; i >= 0; i--)
			if (largestSimpleTrapState[i]) {
				data.push_back(stateList[largestSimpleTrapState[i]].words);
				dataStates.push_back(largestSimpleTrapState[i]);
			}

		for (int i=WordLen-1; i >= 0; i--)
			if (largestComplexTrapState[i]) {
				data.push_back(stateList[largestComplexTrapState[i]].words);
				dataStates.push_back(largestComplexTrapState[i]);
			}

		int numLines = 0;
		for (int i=0; i<data.size(); i++) {
			secrets.push_back(wordsList[data[i].back()].tok);
			numLines = std::max(numLines, int(data[i].size()));
		}

		for (int k=0; k<numLines; k++) {
			int nspaces = 2;
			pr(std::format("  {:2}:", k+1));
			for (int i=0; i<data.size(); i++) {
				if (k >= data[i].size()) {
					nspaces += WordLen + 2;
					continue;
				}
				const Tok &hint = wordsList[data[i][k]].tok;
				for (int j=0; j<nspaces; j++)
					pr(" ");
				prTok(Tok(hint.data(), secrets[i].data()));
				secrets[i] = hint;
				nspaces = 2;
			}
			prNl();
		}
		pr("Paths to these trap states:\n");
		for (int idx : dataStates) {
			pr("  ");
			pr(getPath(idx));
			prNl();
		}
	}

	void doMinMaxSweep()
	{
		vPr(std::format("Mark remaining {} queued states as terminal.\n", stateQueue.size()));
		while (!stateQueue.empty()) {
			terminalStates.push_back(stateQueue.top().second);
			stateQueue.pop();
		}

		int cntEstTerm = 0;
		for (auto idx : terminalStates) {
			auto &state = stateList[idx];
			if (state.depth != 0)
				continue;
			state.depth = state.words.size();
			cntEstTerm++;
		}

		vPr(std::format("Estimated depth for {}/{} terminal states.\n",
				cntEstTerm, terminalStates.size()));

		vPr(std::format("Min-max sweep over {} non-terminal states.\n", nonTerminalStates.size()));
		// iterate over non-terminal states from smalles to largest
		for (auto idx : nonTerminalStates | std::views::reverse) {
			auto &state = stateList[idx];
			state.depth = stateList.size();
			// min-max sweep
			for (auto &guess : state.children) {
				if (guess.empty())
					continue;
				int maxDepth = 0;
				for (int k : guess)
					maxDepth = std::max(maxDepth, stateList[k].depth);
				state.depth = std::min(state.depth, maxDepth+1);
			}
		}
		pr(std::format("Maximum depth at optimal play: {}\n", stateList[0].depth));
	}

	std::vector<std::pair<int,int>> getTrace(int curState = 0)
	{
		std::vector<std::pair<int,int>> traceData;

		while (1)
		{
			auto &state = stateList[curState];
			if (state.children.empty()) {
				for (auto &guess : state.words)
					traceData.emplace_back(guess, curState);
				break;
			}
			std::vector<std::pair<int, int>> choices;
			for (int i = 0; i < state.children.size(); i++) {
				if (state.children.empty())
					continue;
				int maxDepth = 0;
				for (int k : state.children[i])
					maxDepth = std::max(maxDepth, stateList[k].depth);
				if (state.depth != maxDepth+1)
					continue;
				for (int k : state.children[i])
					if (state.depth == stateList[k].depth+1)
						choices.emplace_back(state.words[i], k);
			}
			if (choices.empty())
				break;

			auto &choice = choices[rng(choices.size())];
			traceData.emplace_back(choice.first, curState);
			curState = choice.second;
		}

		return traceData;
	}

	void doTrace()
	{
		std::vector<std::vector<std::pair<int,int>>> traceData;
		for (int i = 0; i < 10; i++)
			traceData.push_back(getTrace());
		pr("Example Traces:\n");
		for (int k = 0; k < traceData.front().size(); k++) {
			pr(std::format("  {:2}:", k+1));
			for (int i = 0; i < traceData.size(); i++) {
				auto *state = &stateList[traceData[i][k].second];
				bool isTerm = state->children.empty();
				pr(isTerm ? " (" : "  ");
				const Tok &hint = wordsList[traceData[i][k].first].tok;
				const Tok &secret = wordsList[traceData[i].back().first].tok;
				prTok(Tok(hint.data(), secret.data()));
				pr(isTerm ? ")" : " ");
			}
			auto *state = &stateList[traceData.back()[k].second];
			pr(std::format(" {:5}\n", state->words.size()));
		}
	}

	void doWriteDatFileLine(std::ofstream &f, int idx)
	{
		const auto &state = stateList[idx];
		for (int i = 1, j = 0; i < wordsList.size(); i++) {
			bool val = j < state.words.size() && state.words[j] == i;
			f << char('0' + val);
			j += val;
		}
		f << char('A' + state.depth) << char('\n');
	}

	const char *vGetShortName() const override { return "minmax"; }

	bool vExecuteCommand(const char *p, const char *arg,
			AbstractWordleDroidEngine *&nextEngine) override
	{
		using namespace std::string_literals;

		if (p == "-minmax"s) {
			return true;
		}

		if (p == "-minmax-q"s) {
			verbose = false;
			return true;
		}

		if (p == "+first"s || p == "+firstGuess"s) {
			firstGuessIdx = findWord(arg);
			return true;
		}

		if (p == "+minStateSize"s) {
			minStateSize = intArg(arg);
			return true;
		}

		if (p == "+maxNumStates"s) {
			maxNumStates = intArg(arg);
			return true;
		}

		if (p == "+maxSearchDepth"s) {
			maxSearchDepth = intArg(arg);
			return true;
		}

		if (p == "+maxTraceLength"s) {
			maxTraceLength = intArg(arg);
			return true;
		}

		if (p == "+limit"s) {
			depthSizeLimits.push_back(intArg(arg));
			return true;
		}

		if (p == "+setup"s) {
			doSetup();
			pr("Best first gusses based on first-level child state sizes:\n");
			doPrintBestGuessesByChildSize(0, 50);
			return true;
		}

		if (p == "+go"s) {
			doSetup();
			while (doBatch()) { }
			if (verbose)
				doShowTraps();
			doMinMaxSweep();
			doTrace();
			return true;
		}

		if (p == "+wrDatFile"s) {
			std::ofstream f(arg ? arg : "wdroid.out");
			if (!f.is_open()) {
				pr("Error: Unable to open file for writing.\n");
				return true;
			}
			std::vector<std::vector<int>> statesBySize;
			statesBySize.resize(stateList[0].depth+2);
			auto addStateToStatesBySize = [&](int idx) {
				const auto &state = stateList[idx];
				if (0 < state.depth && state.depth-1 < statesBySize.size())
					statesBySize[state.depth-1].push_back(idx);
			};
			for (int idx : nonTerminalStates)
				addStateToStatesBySize(idx);
			for (int idx : trapStates)
				addStateToStatesBySize(idx);
			int maxBucketSize = 0;
			for (const auto &item : statesBySize)
				maxBucketSize = std::max(maxBucketSize, int(item.size()));
			for (int i = 0; i < std::min(maxBucketSize, int(25000000 / wordsList.size())); i++)
				for (int j = 0; j < std::min(25, int(statesBySize.size())); j++)
					doWriteDatFileLine(f, statesBySize[j][rng(statesBySize[j].size())]);
			return true;
		}

		return false;
	}
};

REG_WDROID_CMDS(WordleDroidMinMax, "-minmax", "-minmax-q")
