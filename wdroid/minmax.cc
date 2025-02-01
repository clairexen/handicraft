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
	const char *vGetShortName() const override { return "minmax"; }

	using Base = WordleDroidEngine<WordLen>;
	using Base::pr;
	using Base::prFlush;
	using typename Base::Tok;
	using typename Base::WordMsk;
	using Base::wordsList;
	using Base::filterWords;

	struct StateData
	{
		int idx;
		WordMsk msk;
		std::vector<int> words;

		// children[guessWordIdx] = { unique_sorted_next_states }
		std::vector<std::vector<int>> children;
	};

	int minStateSize = 0;
	int maxNumStates = 1000000;

	std::vector<StateData> stateList;
	std::map<WordMsk, int> stateIndex;
	std::priority_queue<std::pair<int, int>> stateQueue;
	std::vector<std::vector<WordMsk>> hintMskTab;

	int getStateIdx(const std::vector<int> &srcWords, const WordMsk &srcMsk)
	{
		if (auto it = stateIndex.find(srcMsk); it != stateIndex.end())
			return it->second;

		auto [wl, msk] = filterWords(srcWords, srcMsk);

		if (auto it = stateIndex.find(msk); it != stateIndex.end()) {
			int idx = it->second;
			stateIndex[srcMsk] = idx;
			return idx;
		}

		int idx = stateList.size();
		stateList.emplace_back(idx, msk, wl);
		stateIndex[srcMsk] = idx;
		stateIndex[msk] = idx;
		stateQueue.emplace(wl.size(), idx);
		return idx;
	}

	void expandState(int idx, int guessWordIdx = 0)
	{
		auto src = [this,idx]() -> auto& { return stateList[idx]; };
		src().children.resize(src().words.size());

		for (int i = 0; i < src().words.size(); i++)
		{
			int ki = src().words[i];
			if (guessWordIdx && guessWordIdx != ki)
				continue;

			std::vector<int> vec;
			for (int j = 0; j < src().words.size(); j++) {
				int kj = src().words[j];
				const WordMsk &msk = hintMskTab[ki][kj];
				vec.push_back(getStateIdx(src().words, msk));
			}

			std::ranges::sort(vec);
			auto subrange = std::ranges::unique(vec);
			vec.erase(subrange.begin(), subrange.end());
			std::swap(src().children[i], vec);
		}
	}

	WordleDroidMinMax(Base *parent) : Base(parent)
	{
		pr("Creating hintMskTab...\n");
		hintMskTab.resize(wordsList.size());
		for (int i = 1; i < wordsList.size(); i++) {
			const char *p = wordsList[i].tok.begin();
			hintMskTab[i].resize(wordsList.size());
			for (int j = 1; j < wordsList.size(); j++) {
				const char *q = wordsList[j].tok.begin();
				hintMskTab[i][j] = Tok(p, q);
			}
		}

		pr("Creating root state...\n");
		std::vector<int> rootWordsList;
		rootWordsList.reserve(wordsList.size());
		for (int i = 1; i < wordsList.size(); i++)
			rootWordsList.push_back(i);
		assert(getStateIdx(rootWordsList, WordMsk::fullMsk()) == 0);
		if (!stateQueue.empty())
			pr(std::format("  size of largest queued state: {:5}\n", stateQueue.top().first));
		pr(std::format("  total number of (new) states: {:5}\n", stateList.size()));

		pr("Creating all first-level child states...\n");
		int prevNumStates = stateList.size();
		assert(stateQueue.top().second == 0);
		stateQueue.pop();
		expandState(0);
		if (!stateQueue.empty())
			pr(std::format("  size of largest queued state: {:5}\n", stateQueue.top().first));
		pr(std::format("  number of new states found: {:7}\n", stateList.size()-prevNumStates));
		pr(std::format("  number of processed states: {:7}\n", 1));
		pr(std::format("  number of queued states:  {:9}\n", stateQueue.size()));
		pr(std::format("  total number of states:   {:9}\n", stateList.size()));
	}

	bool processBatch()
	{
		int batchSize = 10000;
		if (!stateQueue.empty()) {
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
		pr(std::format("Processing next batch of up to {} states...\n", batchSize));
		int i, prevNumStates = stateList.size();
		for (i=0; i<batchSize; i++) {
			if (stateQueue.empty()) break;
			if (stateQueue.top().first < minStateSize) break;
			if (stateList.size() >= maxNumStates) break;
			int idx = stateQueue.top().second;
			stateQueue.pop();
			expandState(idx);
		}
		if (!stateQueue.empty())
			pr(std::format("  size of largest queued state: {:5}\n", stateQueue.top().first));
		pr(std::format("  number of new states found: {:7}\n", stateList.size()-prevNumStates));
		pr(std::format("  number of processed states: {:7}\n", i));
		pr(std::format("  number of queued states:  {:9}\n", stateQueue.size()));
		pr(std::format("  total number of states:   {:9}\n", stateList.size()));
		return i == batchSize;
	}

	bool vExecuteCommand(const char *p, const char *arg,
			AbstractWordleDroidEngine *&nextEngine) override
	{
		using namespace std::string_literals;

		if (p == "-minmax"s) {
			while (processBatch()) { }
			return true;
		}

		return false;
	}
};

REG_WDROID_CMDS(WordleDroidMinMax, "-minmax")
