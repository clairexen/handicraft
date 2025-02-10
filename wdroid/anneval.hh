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

#ifndef ANNEVAL_HH
#define ANNEVAL_HH

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>

struct WordleDroidAnnEval
{
	int inputDim = 0, innerDim = 0;
	std::vector<std::vector<float>> inputWeights;
	std::vector<float> innerBias;
	std::vector<float> outputWeights;
	float outputBias = 0.0;

	bool okay = false;
	mutable std::vector<float> innerValues; // buffer for evalModel()

	inline operator bool() const { return okay; }
	bool readModelBinFile(const std::string &fn);
	float evalModel(const std::vector<int> &enabledInputs) const;
	void clear();
};

#endif
