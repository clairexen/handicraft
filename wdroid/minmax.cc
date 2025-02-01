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

	std::vector<StateData> stateList;
	std::map<WordMsk, int> stateIndex;
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
		return idx;
	}

	void expandState(int idx)
	{
		auto src = [this,idx]() -> auto& { return stateList[idx]; };
		src().children.resize(src().words.size());

		for (int i = 0; i < src().words.size(); i++)
		{
			std::vector<int> vec;

			int ki = src().words[i];
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
		pr(std::format("  number of states in database: {}\n", stateList.size()));

		pr("Creating depth=1 states...\n");
		// int cur1 = stateList.size();
		expandState(0);
		pr(std::format("  number of states in database: {}\n", stateList.size()));

	#if 0
		pr("Creating depth=2 states...\n");
		int cur2 = stateList.size();
		for (int i = cur1; i < cur2; i++)
			expandState(i);
		pr(std::format("  number of states in database: {}\n", stateList.size()));
	#endif
	}

	bool vExecuteCommand(const char *p, const char *arg,
			AbstractWordleDroidEngine *&nextEngine) override
	{
		using namespace std::string_literals;

		if (p == "-minmax"s) {
			pr(".. minmax ..\n");
			return true;
		}

		return false;
	}
};

REG_WDROID_CMDS(WordleDroidMinMax, "-minmax")
