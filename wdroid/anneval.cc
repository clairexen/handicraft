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

#undef DEBUG_ANN_TEST

bool WordleDroidAnnEval::readModelBinFile(const std::string &fn)
{
	okay = false;

	std::ifstream file(fn, std::ios::binary);
	if (!file.is_open()) return false;

	file.read(reinterpret_cast<char*>(&inputDim), sizeof(int));
	file.read(reinterpret_cast<char*>(&innerDim), sizeof(int));
	if (!file) return false;

#ifdef DEBUG_ANN_TEST
	std::cerr << "Dimensions: " << inputDim << " " << innerDim << std::endl;
#endif

	inputWeights.resize(inputDim);
	for (int i = 0; i < inputDim; i++) {
		inputWeights[i].resize(innerDim);
		file.read(reinterpret_cast<char*>(inputWeights[i].data()), innerDim * sizeof(float));
		if (!file) return false;
	}

#ifdef DEBUG_ANN_TEST
	std::cerr << "First weight vector:";
	for (int i = 0; i < 5 && i < innerDim; i++)
		std::cerr << " " << inputWeights[0][i];
	std::cerr << " ..." << std::endl;
#endif

	innerBias.resize(innerDim);
	file.read(reinterpret_cast<char*>(innerBias.data()), innerDim * sizeof(float));
	if (!file) return false;

#ifdef DEBUG_ANN_TEST
	std::cerr << "Inner bias vector:";
	for (int i = 0; i < 5 && i < innerDim; i++)
		std::cerr << " " << innerBias[i];
	std::cerr << " ..." << std::endl;
#endif

	outputWeights.resize(innerDim);
	file.read(reinterpret_cast<char*>(outputWeights.data()), innerDim * sizeof(float));
	if (!file) return false;

#ifdef DEBUG_ANN_TEST
	std::cerr << "Output weights vector:";
	for (int i = 0; i < 5 && i < innerDim; i++)
		std::cerr << " " << outputWeights[i];
	std::cerr << " ..." << std::endl;
#endif

	file.read(reinterpret_cast<char*>(&outputBias), sizeof(float));
	if (!file) return false;

#ifdef DEBUG_ANN_TEST
	std::cerr << "Output bias: " << outputBias << std::endl;
#endif

	int k;
	file.read(reinterpret_cast<char*>(&k), sizeof(int));
	if (!file) return false;
	testInput.resize(k);
	file.read(reinterpret_cast<char*>(testInput.data()), k * sizeof(int));
	if (!file) return false;

	testInner.resize(innerDim);
	file.read(reinterpret_cast<char*>(testInner.data()), innerDim * sizeof(float));
	if (!file) return false;

	testReLU.resize(innerDim);
	file.read(reinterpret_cast<char*>(testReLU.data()), innerDim * sizeof(float));
	if (!file) return false;

	file.read(reinterpret_cast<char*>(&testUnscaledOut), sizeof(float));
	if (!file) return false;

	file.read(reinterpret_cast<char*>(&testScaledOut), sizeof(float));
	if (!file) return false;

	// expect EOF
	char c;
	file.read(&c, 1);
	if (!file.eof()) return false;

	innerValues.resize(innerDim);
	okay = true;

	float testOut = evalModel(testInput);
	assert(fabsf(testOut - testScaledOut) < 0.01);
#ifdef DEBUG_ANN_TEST
	assert(!"DEBUG_ANN_TEST is defined but evalModel() returns correct test value!");
#endif
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

#ifdef DEBUG_ANN_TEST
	std::cerr << "Comparing innerValues and ReLU outputs:\n";
	for (int i = 0, k = 0; i < innerDim && k < 10; i++) {
		float reLuVal = innerValues[i] > 0 ? innerValues[i] : 0;
		if (i > 3 && fabsf(innerValues[i] - testInner[i]) < 0.01 &&
				fabsf(reLuVal - testReLU[i]) < 0.01)
			continue;
		std::cerr << i << ": " << innerValues[i] << " " << testInner[i]
				<< ", " << reLuVal << " " << testReLU[i] << std::endl;
		if (k++ == 9)
			std::cerr << "..." << std::endl;
	}
#endif

	for (int i = 0; i < innerDim; i++)
		if (innerValues[i] > 0)
			outputValue += outputWeights[i] * innerValues[i];

#ifdef DEBUG_ANN_TEST
	std::cerr << "Comparing unscaled output value: " << outputValue
			<< " " << testUnscaledOut << std::endl;
#endif

	outputValue = outputValue * (8.0 - 1.0) + 1.0;

#ifdef DEBUG_ANN_TEST
	std::cerr << "Comparing scaled output value: " << outputValue
			<< " " << testScaledOut << std::endl;
#endif

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
