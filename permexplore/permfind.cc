// Copyright (C) 2019  Clifford Wolf <clifford@clifford.at>
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

// Find permutation sequences using ROR/GREV/SHFL by creating a database
// of permutations reachable in N instructions, then scan for 2N sequences
// for given permutations.
//
// Just as an example, this searches for bitswaps and bitfield rotate shifts.
// Modify as needed.

#include "permdb.h"

#define CONFIG_N  32 // number of chunks
#define CONFIG_SZ  1 // bits in a chunk
#define MAXWAVES   3 // maximum depth

typedef pdb::perm_t<CONFIG_N, CONFIG_SZ> perm_t;
typedef pdb::permdb_t<CONFIG_N, CONFIG_SZ> permdb_t;
typedef pdb::GenericMiner<CONFIG_N, CONFIG_SZ> GenericMiner;

struct BitSwapMiner
{
	std::pair<int,std::string> map[CONFIG_N][CONFIG_N];

	BitSwapMiner()
	{
		for (int i = 0; i < CONFIG_N; i++)
		for (int j = 0; j < CONFIG_N; j++)
			map[i][j].first = -1;
	}

	void wave(const permdb_t &database)
	{
		printf("BitSwapMiner:\n");
		for (int i = 0; i < CONFIG_N; i++)
		{
			for (int j = 0; j <= i; j++)
			{
				if (map[i][j].first < 0) {
					perm_t needle = perm_t::identity();
					needle.swap(i, j);
					map[i][j] = database.find2(needle, pdb::stringf("swap(%d,%d)", i, j));
				}
			}
			printf("=== swap(%d,*):", i);
			for (int l = 0; l <= i; l++)
				if (map[i][l].first < 0)
					printf("  -");
				else
					printf("%3d", map[i][l].first);
			printf(" ===\n");
		}
		printf("--------\n");
		fflush(stdout);
	}

	void summary()
	{
		printf("BitSwapMiner:\n");
		for (int k = 0; k < CONFIG_N; k++) {
			for (int l = 0; l <= k; l++)
				if (map[k][l].first < 0)
					printf("  -");
				else
					printf("%3d", map[k][l].first);
			printf("\n");
		}
		for (int i = 0; i < CONFIG_N; i++)
		for (int j = 0; j <= i; j++)
			if (map[i][j].first == 0)
				printf("Swapping bits %d and %d: identity\n", i, j);
			else if (map[i][j].first > 0)
				printf("Swapping bits %d and %d: %s\n", i, j, map[i][j].second.c_str());
			else
				printf("Swapping bits %d and %d: ** not found **\n", i, j);
		printf("--------\n");
		fflush(stdout);
	}
};

int main()
{
	permdb_t database;
	database.insert(perm_t::identity());

	BitSwapMiner bitswapminer;
	GenericMiner genericminer;

	// rotate arbitrary bit-fields
	for (int start = 0; start < CONFIG_N; start++)
	for (int stop = start; stop < CONFIG_N; stop++)
	{
		perm_t needle = perm_t::identity();
		for (int i = start; i < stop; i++)
			needle.set(i+1, i);
		needle.set(start, stop);
		genericminer.add(needle, pdb::stringf("rot(%d:%d)", stop, start));
	}


	for (int wavecnt = 1; !database.queue.empty() && wavecnt <= MAXWAVES; wavecnt++)
	{
		database.wave(pdb::stringf("Wave #%d", wavecnt));
		if (wavecnt > 2) {
			bitswapminer.wave(database);
			genericminer.wave(database);
		}
	}

	bitswapminer.summary();
	genericminer.summary();

	return 0;
}
