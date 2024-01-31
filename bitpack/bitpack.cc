/*
 *  Generate bit sequences with interesting properties..
 *
 *  Copyright (C) 2012  RIEGL Research ForschungsGmbH
 *  Copyright (C) 2012  Clifford Wolf <clifford@clifford.at>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */

// Retulst for LEN=3 to LEN=127
//
// 110
// 1110100
// 111101011001000
// 1111100110100100001010111011000
// 1111101100111000011010100100010
// 1111101110001010110100001100100
// 111111010000011100001001000110110010110101110111100110001010100
// 111111010101100110111011010010011100010111100101000110000100000
// 111111011010001000010110010101001001111000001101110011000111010
// 1111111000111011000101001011111010101000010110111100111001010110011000001101101011101000110010001000000100100110100111101110000
// 1111111001000000101100000111010000100111000110100100101110110111001101100101011010111110111100001100010001010011001111010101000
// 1111111010010010101110101010011100011001011001100010111101110010001000011010110100011110000010011011000010100000011101101111100
// 1111111010101001100111011101001011000110111101101011011001001000111000010111110010101110011010001001111000101000011000001000000
// 1111111010110111011110001110100010101110000001111011001100010010011100111110010000010001101010100110110100101000010110000110010
// 1111111011010100000011011111000001011000010000111010001100010011100101001101001011110101110111000111100110010010001010101101100
// 1111111011100111101000001010111100100100010000100111011010101001011101011000000110010100011100001111100010110110011000110100110
// 1111111011101101111010001011001011111000100000011001101100011100111010111000010011000001010101101001001010011110010001101010000
// 1111111011111000111010101001010111101001100111001101011000100010111011001000011110010110111000001010001101101000000110000100100

#include <stdint.h>
#include <stdio.h>
#include <algorithm>
#include <bitset>
#include <vector>
#include <map>

#ifndef LEN
#  error Missing -DLEN= compiler option
#endif

#define USE_BLOCK_PREFIX
#define QUICK_LIMIT 6

typedef std::bitset<LEN> bitvect;

#define ONE bitvect(1)
#define FULL (~bitvect(0))

static bitvect shift(bitvect v, int idx)
{
	return (v << idx) | (v >> (LEN-idx));
}

static bitvect mirror(bitvect v)
{
	bitvect u = 0;
	for (int i = 0; i < LEN; i++) {
		u = (u << 1) | (v & ONE);
		v = v >> 1;
	}
	return u;
}

struct pattern_s
{
	int pos;
	bitvect bits, mask;
	bitvect shifted_bits[LEN];
	bitvect shifted_mask[LEN];
	pattern_s(bitvect b = 0, bitvect m = 0) : pos(-1), bits(b), mask(m) {
		for (int i = 0; i < LEN; i++) {
			shifted_bits[i] = shift(bits, i);
			shifted_mask[i] = shift(mask, i);
		}
	}
};

static bool comp_pattern_len(const pattern_s &a, const pattern_s &b)
{
	return a.mask.count() > b.mask.count();
}

static void print_state(std::vector<pattern_s> &state, bitvect bitdata, bitvect mask)
{
	printf("\nState: ");
	for (int i = LEN-1; i >= 0; i--)
		printf("%c", mask.test(i) ? (bitdata.test(i) ? '1' : '0') : 'x');
	for (size_t i = 0; i < state.size(); i++) {
		printf("\n%5zd: ", i);
		pattern_s &p = state[i];
		bitvect m = shift(p.mask, p.pos);
		bitvect b = shift(p.bits, p.pos);
		for (int j = LEN-1; j >= 0; j--) {
			printf("%c", m.test(j) ? (b.test(j) ? '1' : '0') : '.');
		}
	}
	printf("\n");
}

static bool bitpack_worker(std::vector<pattern_s> &state, bitvect bitdata, bitvect mask, int depth, bool verbose)
{
	if (depth == int(state.size())) {
		if (verbose)
			print_state(state, bitdata, mask);
		return true;
	}

	pattern_s &p = state[depth];
	for (p.pos = 0; p.pos < LEN; p.pos++) {
		bitvect m = mask & p.shifted_mask[p.pos];
		if ((bitdata & m) == (p.shifted_bits[p.pos] & m)) {
			if (bitpack_worker(state, bitdata | p.shifted_bits[p.pos],
					mask | p.shifted_mask[p.pos], depth+1, verbose))
				return true;
		}
	}

	return false;
}

bool bitpack(std::vector<pattern_s> &state, bool verbose)
{
	std::sort(state.begin(), state.end(), comp_pattern_len);
	state[0].pos = 0;
	return bitpack_worker(state, state[0].bits, state[0].mask, 1, verbose);
}

bool check_prefix(bitvect prefix, int depth, bool quick, bool verbose)
{
	std::vector<pattern_s> state;
	state.reserve(depth);

	bitvect buf = prefix;
	bitvect mask = ~(FULL << depth);

	for (int i = 0; i < depth; i++) {
		for (size_t j = 0; j < state.size(); j++)
			for (int k = 0; k < LEN; k++) {
				if (((state[j].mask >> k) & mask) != mask)
					break;
				if (((state[j].bits >> k) & mask) == buf)
					goto next_i;
			}
#ifdef QUICK_LIMIT
		if (quick && state.size() > QUICK_LIMIT)
			break;
#endif
		state.push_back(pattern_s(buf, mask));
	next_i:;
		prefix = prefix >> 1;
		mask = mask >> 1;
		buf = (buf ^ prefix) & mask;
	}

	return bitpack(state, verbose);
}

void next_stage(std::vector<bitvect> &prefixes, int &depth)
{
	std::vector<bitvect> new_prefixes;

	printf("\nChecking %zd prefixes of length %d/%d:\n%4d: ", prefixes.size(), depth, LEN, depth);

	time_t begin_time = time(NULL);
	for (size_t i = 0; i < prefixes.size(); i++) {
		if (prefixes[i].count() > LEN/2+1 || (prefixes[i] ^ ~(FULL << depth)).count() > LEN/2) {
			putchar(',');
		} else if (check_prefix(prefixes[i], depth, depth < LEN/2, false)) {
			new_prefixes.push_back(prefixes[i]);
			new_prefixes.push_back(prefixes[i] | (ONE << depth));
			putchar(depth < LEN/2 ? 'z' : 'x');
		} else
			putchar('.');
		if (i % 100 == 99 || i == prefixes.size()-1) {
			if (i == prefixes.size()-1) {
				for (int j = i; j % 100 != 99; j++)
					putchar(' ');
				printf(" [done 100.00%%,");
			} else
				printf(" [done %6.2f%%,", i * 100.0 / prefixes.size());
			if (prefixes.size() > 1000) {
				int eta = double(prefixes.size() - i) * double(time(NULL) - begin_time) / double(i);
				printf(" passed %6.2f%%, ETA %2d:%02d:%02d]\n", new_prefixes.size() * 50.0 / (i+1),
						eta / 60 / 60, (eta / 60) % 60, eta % 60);
			} else
				printf(" passed %6.2f%%]\n", new_prefixes.size() * 50.0 / (i+1));
			if (i < prefixes.size()-1)
				printf("%4d: ", depth);
		}
		fflush(stdout);
	}

	printf("\nGenerated %zd new prefixes of length %d:\n", new_prefixes.size(), depth);
	for (size_t k = 0; k < new_prefixes.size(); k++) {
		printf("%4zd: ", k+1);
		for (int i = depth-1; i >= 0; i--)
			printf("%c", new_prefixes[k].test(i) ? '1' : '0');
		printf("\n");
	}

	prefixes.swap(new_prefixes);
	depth++;
}

#if 0
void test_bitvect_type()
{
	printf("\nPrimitive bit operations tests (size = %zd bytes):\n\n", sizeof(bitvect));

	bitvect v = ONE;
	for (int k = 0; k < LEN; k++) {
		printf("%4d: ", k);
		for (int i = LEN-1; i >= 0; i--)
			printf("%c", v.test(i) ? '1' : '.');
		printf(" 0x");
		for (int i = sizeof(bitvect); i >= 0; i--)
			printf("%02x", ((unsigned char*)&v)[i]);
		printf("\n");
		v = shift(v, 1);
	}

	printf("\n");
	for (int k = 0; k < LEN; k++) {
		v = shift(0b1001, k);
		printf("%4d: ", k);
		for (int i = LEN-1; i >= 0; i--)
			printf("%c", v.test(i) ? '1' : '.');
		printf(" 0x");
		for (int i = sizeof(bitvect); i >= 0; i--)
			printf("%02x", ((unsigned char*)&v)[i]);
		printf("\n");
	}
}
#endif

int main()
{
	std::vector<bitvect> prefixes;

#if LEN == 3
	int depth = 3;
#elif LEN == 7
	int depth = 6;
#elif LEN == 15
	int depth = 8;
#elif LEN == 31
	int depth = 12;
#elif LEN == 63
	int depth = 16;
#elif LEN == 127
	int depth = 26;
#else
#  error No initial depth configured for this length
#endif

#ifdef USE_BLOCK_PREFIX
	int block_increment = 2, block_prefix = 0;
	for (int i = 1; i < LEN; i = i*2) {
		block_prefix = (block_prefix << 1) | 1;
		block_increment = block_increment << 1;
	}
	block_prefix = block_prefix << 1;
	for (int i = 0; i < (1 << depth); i += block_increment)
		prefixes.push_back(i | block_prefix);
#else
	for (int i = 0; i < (1 << depth); i++)
		prefixes.push_back(i);
#endif

	while (depth < LEN)
		next_stage(prefixes, depth);

	std::map<std::string,bitvect> results;

	for (size_t i = 0; i < prefixes.size(); i++)
	{
		if (prefixes[i].count() != LEN/2+1)
			continue;
		check_prefix(prefixes[i], LEN, false, true);

		bitvect v = prefixes[i];
		for (int j = 0; j < LEN; j++) {
			bitvect u = shift(prefixes[i], j);
			if (u.to_string() > v.to_string()) v = u;
			u = shift(mirror(prefixes[i]), j);
			if (u.to_string() > v.to_string()) v = u;
		}
		results[v.to_string()] = v;
	}

	for (std::map<std::string,bitvect>::iterator it = results.begin(); it != results.end(); it++)
	{
		bitvect v = it->second;
		printf("%sCode: ", it == results.begin() ? "\n" : "");
		for (int i = LEN-1; i >= 0; i--)
			printf("%c", v.test(i) ? '1' : '0');
		printf("\n");
	}

	printf("\nREADY.\n");
	return 0;
}

