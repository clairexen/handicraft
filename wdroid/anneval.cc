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

#include "anneval.hh"

bool WordleDroidAnnEval::readModelBinFile(const std::string &fn)
{
	okay = false;

	std::ifstream file(fn, std::ios::binary);
	if (!file.is_open()) return false;

	file.read(reinterpret_cast<char*>(&inputDim), sizeof(int));
	file.read(reinterpret_cast<char*>(&innerDim), sizeof(int));
	if (!file) return false;

	inputWeights.resize(inputDim);
	for (int i = 0; i < inputDim; i++) {
		inputWeights[i].resize(innerDim);
		file.read(reinterpret_cast<char*>(inputWeights[i].data()), innerDim*sizeof(float));
		if (!file) return false;
	}

	innerBias.resize(innerDim);
	file.read(reinterpret_cast<char*>(innerBias.data()), innerDim*sizeof(float));
	if (!file) return false;

	outputWeights.resize(innerDim);
	file.read(reinterpret_cast<char*>(outputWeights.data()), innerDim*sizeof(float));
	if (!file) return false;

	file.read(reinterpret_cast<char*>(&outputBias), sizeof(float));
	if (!file) return false;

	// expect EOF
	char c;
	file.read(&c, 1);
	if (!file.eof()) return false;

	innerValues.resize(innerDim);
	okay = false;

	return true;
}

float WordleDroidAnnEval::evalModel(const std::vector<int> &enabledInputs) const
{
	float outputValue = outputBias;
	assert(okay);

	for (int i = 0; i < innerDim; i++)
		innerValues[i] = innerBias[i];

	for (int idx : enabledInputs) {
		const float *w = inputWeights[idx-1].data();
		for (int i = 0; i < innerDim; i++)
			innerValues[i] += w[i];
	}

	for (int i = 0; i < innerDim; i++)
		if (innerValues[i] > 0)
			outputValue += outputWeights[i] * innerValues[i];

	outputValue = outputValue * (8.0 - 1.0) + 1.0;
	return outputValue < 0.5 ? 0.5 : outputValue;
}

void WordleDroidAnnEval::clear()
{
	okay = false;
	inputDim = 0;
	innerDim = 0;
	inputWeights.clear();
	innerBias.clear();
	outputWeights.clear();
	outputBias = 0.0;
	innerValues.clear();
}
