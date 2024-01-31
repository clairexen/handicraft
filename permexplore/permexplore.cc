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

// Brute-force the number of bit permutations reachable using ROR/GREV/SHFL
// operations.

#include "permdb.h"
#include <algorithm>

#define CONFIG_N  32 // number of chunks
#define CONFIG_SZ  1 // bits in a chunk
#define MAXWAVES   4 // maximum depth

#define FINDSWAPS
#undef PRINTALL

typedef pdb::perm_t<CONFIG_N, CONFIG_SZ> perm_t;
typedef pdb::permdb_t<CONFIG_N, CONFIG_SZ> permdb_t;

int main()
{
	permdb_t database;
	database.insert(perm_t::identity());

	std::vector<size_t> nperms;
	nperms.push_back(database.size());

	while (!database.queue.empty() && nperms.size() <= MAXWAVES) {
		database.wave(pdb::stringf("Wave #%d", int(nperms.size())));
		nperms.push_back(database.size());
	}

	printf("\n");
	printf("-- reachable permutations --\n");
	for (int i = 0; i < int(nperms.size()); i++)
		printf("after %2d instructions: %10ld\n", i, long(nperms[i]));
	printf("----------------------------\n");

#ifdef FINDSWAPS
	{
		perm_t p;

		printf("\n");
		printf("-- permutations of interest --\n");

		p = perm_t::identity();
		p.swap(0, 1);
		database.find2(p, "lsbswap(1,0)");
		database.find2(p.grev(-1), "msbswap(1,0)");

		printf("----\n");

		p = perm_t::identity();
		p.swap(0, 4);
		p.swap(1, 5);
		p.swap(2, 6);
		p.swap(3, 7);
		database.find2(p, "lsbswap(4,0)");
		database.find2(p.grev(-1), "msbswap(4,0)");

		printf("----\n");

		p = perm_t::identity();
		p.swap(0, 8);
		p.swap(1, 9);
		p.swap(2, 10);
		p.swap(3, 11);
		database.find2(p, "lsbswap(4,4)");
		database.find2(p.grev(-1), "msbswap(4,4)");

		printf("----\n");

		p = perm_t::identity();
		p.swap(0, 12);
		p.swap(1, 13);
		p.swap(2, 14);
		p.swap(3, 15);
		database.find2(p, "lsbswap(4,8)");
		database.find2(p.grev(-1), "msbswap(4,8)");

		printf("------------------------------\n");
		fflush(stdout);
	}
#endif

#ifdef PRINTALL
	{
		printf("\n");
		printf("-- all permutations --\n");

		std::vector<perm_t> keys;
		for (long i = 0; i < database.size(); i++)
			keys.push_back(database.perm(i));
		std::sort(keys.begin(), keys.end());
		for (auto &key : keys)
			printf("%s\n", key.str().c_str());
		printf("-------------------\n");
		fflush(stdout);
	}
#endif

	printf("DONE\n");
	fflush(stdout);
	return 0;
}
