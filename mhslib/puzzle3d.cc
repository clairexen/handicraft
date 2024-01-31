/*
 *  ezSAT -- A simple and easy to use CNF generator for SAT solvers
 *
 *  Copyright (C) 2013  Clifford Wolf <clifford@clifford.at>
 *  
 *  Permission to use, copy, modify, and/or distribute this software for any
 *  purpose with or without fee is hereby granted, provided that the above
 *  copyright notice and this permission notice appear in all copies.
 *  
 *  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 *  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 *  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 *  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 *  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 *  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 */

#include "mhslib.h"
#include <stdio.h>
#include <assert.h>
#include <string>
#include <map>

#define DIM_X 5
#define DIM_Y 5
#define DIM_Z 5

#define NUM_124 6
#define NUM_223 6

MultiHotSolver mhs;
int blockidx = 0, literalIdx = 0;
std::map<int, std::string> blockinfo;
std::vector<int> grid[DIM_X][DIM_Y][DIM_Z];

int add_block(int pos_x, int pos_y, int pos_z, int size_x, int size_y, int size_z, int blockidx)
{
	char buffer[1024];
	snprintf(buffer, 1024, "block(%d,%d,%d,%d,%d,%d,%d);", size_x, size_y, size_z, pos_x, pos_y, pos_z, blockidx);

	int var = literalIdx++;
	blockinfo[var] = buffer;

	for (int ix = pos_x; ix < pos_x+size_x; ix++)
	for (int iy = pos_y; iy < pos_y+size_y; iy++)
	for (int iz = pos_z; iz < pos_z+size_z; iz++)
		grid[ix][iy][iz].push_back(var);

	return var;
}

void add_block_positions_124(std::vector<int> &block_positions_124)
{
	block_positions_124.clear();
	for (int size_x = 1; size_x <= 4; size_x *= 2)
	for (int size_y = 1; size_y <= 4; size_y *= 2)
	for (int size_z = 1; size_z <= 4; size_z *= 2) {
		if (size_x == size_y || size_y == size_z || size_z == size_x)
			continue;
		for (int ix = 0; ix <= DIM_X-size_x; ix++)
		for (int iy = 0; iy <= DIM_Y-size_y; iy++)
		for (int iz = 0; iz <= DIM_Z-size_z; iz++)
			block_positions_124.push_back(add_block(ix, iy, iz, size_x, size_y, size_z, blockidx++));
	}
}

void add_block_positions_223(std::vector<int> &block_positions_223)
{
	block_positions_223.clear();
	for (int orientation = 0; orientation < 3; orientation++) {
		int size_x = orientation == 0 ? 3 : 2;
		int size_y = orientation == 1 ? 3 : 2;
		int size_z = orientation == 2 ? 3 : 2;
		for (int ix = 0; ix <= DIM_X-size_x; ix++)
		for (int iy = 0; iy <= DIM_Y-size_y; iy++)
		for (int iz = 0; iz <= DIM_Z-size_z; iz++)
			block_positions_223.push_back(add_block(ix, iy, iz, size_x, size_y, size_z, blockidx++));
	}
}

int main()
{
	// add 1x2x4 blocks
	std::vector<int> block_positions_124;
#if 0
	for (int i = 0; i < NUM_124; i++) {
		add_block_positions_124(block_positions_124);
		mhs.addClause(block_positions_124);
	}
#else
	add_block_positions_124(block_positions_124);
	mhs.addClause(block_positions_124, NUM_124);
#endif

	// add 2x2x3 blocks
	std::vector<int> block_positions_223;
#if 0
	for (int i = 0; i < NUM_223; i++) {
		add_block_positions_223(block_positions_223);
		mhs.addClause(block_positions_223);
	}
#else
	add_block_positions_223(block_positions_223);
	mhs.addClause(block_positions_223, NUM_223);
#endif

	// add constraint for max one block per grid element
	for (int ix = 0; ix < DIM_X; ix++)
	for (int iy = 0; iy < DIM_Y; iy++)
	for (int iz = 0; iz < DIM_Z; iz++) {
		assert(grid[ix][iy][iz].size() > 0);
		mhs.addClause(grid[ix][iy][iz], 0, 1);
	}

	mhs.print();

	printf("Solving puzzle (this may take a while)..\n");

	std::vector<bool> model;
	bool ok = mhs.solve(model, 20, 1000000, 200);

	if (ok) {
		printf("Puzzle solution:\n");
		for (auto it : blockinfo)
			if (model[it.first])
				printf("%s\n", it.second.c_str());
	} else
		printf("No solution found!\n");

	return 0;
}

