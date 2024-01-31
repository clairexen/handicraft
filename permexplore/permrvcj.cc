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

// Demonstrate the use of repeated indices for "dont-care".
// Example application: Decode RISC-V CJ-type Immediates

#include "permdb.h"
#include <tuple>

#define CONFIG_N  16 // number of chunks
#define CONFIG_SZ  1 // bits in a chunk
#define MAXWAVES   6 // maximum depth

typedef pdb::perm_t<CONFIG_N, CONFIG_SZ> perm_t;
typedef pdb::permdb_t<CONFIG_N, CONFIG_SZ> permdb_t;

int main()
{
	perm_t init;
	perm_t needle;

	init.set(15,  0);
	init.set(14,  0);
	init.set(13,  0);
	init.set(12, 12);
	init.set(11, 11);
	init.set(10, 10);
	init.set( 9,  9);
	init.set( 8,  8);
	init.set( 7,  7);
	init.set( 6,  6);
	init.set( 5,  5);
	init.set( 4,  4);
	init.set( 3,  3);
	init.set( 2,  2);
	init.set( 1,  0);
	init.set( 0,  0);

	// CJ-type immediate
	needle.set(15,  0);
	needle.set(14,  0);
	needle.set(13,  0);
	needle.set(12,  0);
	needle.set(11,  0);
	needle.set(10, 12);
	needle.set( 9,  8);
	needle.set( 8, 10);
	needle.set( 7,  9);
	needle.set( 6,  6);
	needle.set( 5,  7);
	needle.set( 4,  2);
	needle.set( 3, 11);
	needle.set( 2,  5);
	needle.set( 1,  4);
	needle.set( 0,  3);

	permdb_t fwd_database;
	fwd_database.insert(init);

	permdb_t bwd_database;
	bwd_database.insert(needle);

	for (int wavecnt = 1; (!fwd_database.queue.empty() || !bwd_database.queue.empty()) && wavecnt <= MAXWAVES; wavecnt++)
	{
		if (!fwd_database.queue.empty())
			fwd_database.wave(pdb::stringf("Fwd. Wave #%d", wavecnt));
		if (!bwd_database.queue.empty())
			bwd_database.wave(pdb::stringf("Bwd. Wave #%d", wavecnt));
		if (pdb::FwdBwdScan<CONFIG_N,CONFIG_SZ>(fwd_database, bwd_database))
			break;
	}

	return 0;
}
