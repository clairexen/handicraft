// Simple M-Sequence generator.

#include <stdio.h>
#include <stdlib.h>
#include <bitset>
#include <map>

template <int len>
static std::bitset<len> shift(std::bitset<len> v, int idx)
{
	return (v << idx) | (v >> (len-idx));
}

template <int len>
static std::bitset<len> mirror(std::bitset<len> v)
{
	std::bitset<len> u(0);
	for (int i = 0; i < len; i++) {
		u = (u << 1) | (v & std::bitset<len>(1));
		v = v >> 1;
	}
	return u;
}

std::map<std::string, std::string> sequence_cache;

template <int len>
bool gensequence(unsigned long mask = 0, bool verbose = true)
{
	typedef std::bitset<(1 << len)-1> bitdata_t;
	bitdata_t statelog(0);
	bitdata_t bitdata(0);
	unsigned long state = 1;

	if (mask == 0)
	{
		sequence_cache.clear();
		for (mask = 1; mask < (1 << len); mask++)
			gensequence<len>(mask, false);
		for (std::map<std::string, std::string>::iterator it = sequence_cache.begin(); it != sequence_cache.end(); it++)
			printf("%s (%s)\n", it->first.c_str(), it->second.c_str());
		return !sequence_cache.empty();
	}

	for (int i = 0; i < (1 << len)-1; i++) {
		bool newbit = __builtin_popcount(state & mask) % 2 == 1;
		bitdata = (bitdata << 1) | bitdata_t(state & 1);
		state = (state >> 1) | (newbit << (len-1));
		if (state == 0 || statelog[state-1])
			return false;
		statelog[state-1] = 1;
	}

	std::string best = bitdata.to_string();
	for (int i = 0; i < (1 << len)-1; i++) {
		std::string a = shift<(1 << len)-1>(bitdata, i).to_string();
		std::string b = mirror<(1 << len)-1>(shift<(1 << len)-1>(bitdata, i)).to_string();
		if (a > best) best = a;
		if (b > best) best = b;
	}

	if (verbose) {
		printf("%s (%s)\n", best.c_str(), std::bitset<len>(mask).to_string().c_str());
		fflush(stdout);
	}

	if (sequence_cache.count(best))
		sequence_cache[best] += ", " + std::bitset<len>(mask).to_string();
	else
		sequence_cache[best] = std::bitset<len>(mask).to_string();

	return true;
}

int main()
{
	// gensequence< 2>(0b00000011);
	// gensequence< 3>(0b00000011);
	// gensequence< 4>(0b00000011);
	// gensequence< 5>(0b00000101);
	// gensequence< 6>(0b00000011);
	// gensequence< 7>(0b00000011);
	// gensequence< 8>(0b01100011);
	// gensequence< 9>(0b00010001);
	// gensequence<10>(0b00001001);
	// gensequence<11>(0b00000101);
	// gensequence<12>(0b10011001);

	gensequence<2>();
	gensequence<3>();
	gensequence<4>();
	gensequence<5>();
	gensequence<6>();
	gensequence<7>();

	// gensequence<8>();
	// gensequence<9>();
	// gensequence<10>();
	// gensequence<11>();
	// gensequence<12>();
	return 0;
}

