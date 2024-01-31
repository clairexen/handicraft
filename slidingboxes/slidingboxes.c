/*
 *  Brute-force solver for sliding block puzzles
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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#define HASH_SIZE (16*1024*1024-1)

int8_t w, h, np;
int8_t *finish_mask;
int8_t **state_pool;
int *state_hash[HASH_SIZE];
int *edge_infos, *block_infos;
int state_pool_size;
int finish_state;
int search_level;
int cycle_size;

static void *exp_realloc(void *ptr, size_t size)
{
	size_t i = 1;
	while (size >> i)
		size |= size >> i++;
	return realloc(ptr, size+1);
}

static bool make_new(int8_t *to, int8_t *from, int8_t p, int8_t dx, int8_t dy)
{
	int8_t x, y;
	memset(to, 0, w*h);

	for (x = 0; x < w; x++)
	for (y = 0; y < h; y++) {
		int8_t nx = x, ny = y;
		if (from[y*w + x] == 0)
			continue;
		if (from[y*w + x] == p) {
			nx = x + dx, ny = y + dy;
			if (nx < 0 || nx >= w || ny < 0 || ny >= h)
				return false;
		}
		if (to[ny*w + nx] != 0)
			return false;
		to[ny*w + nx] = from[y*w + x];
	}

	return true;
}

static int hash_state(int8_t *state)
{
	int i;
	uint32_t hash = 0;
	for (i = 0; i < w*h; i++)
		hash = state[i] + (hash << 6) + (hash << 16) - hash;
	return hash % HASH_SIZE;
}

static void add_state(int8_t *state, int from, int p)
{
	int i, hash = hash_state(state);
	if (state_hash[hash] != NULL) {
		int *p = state_hash[hash];
		for (i = 1; i <= *p; i++)
			if (!memcmp(state, state_pool[p[i]], w*h))
				return;
	}

	state_pool_size++;
	state_pool = exp_realloc(state_pool, sizeof(int8_t*) * state_pool_size);
	state_pool[state_pool_size-1] = malloc(w*h);
	memcpy(state_pool[state_pool_size-1], state, w*h);

	edge_infos = exp_realloc(edge_infos, sizeof(int*) * state_pool_size);
	edge_infos[state_pool_size-1] = from;

	block_infos = exp_realloc(block_infos, sizeof(int*) * state_pool_size);
	block_infos[state_pool_size-1] = p;

	if (state_hash[hash] != NULL) {
		state_hash[hash][0]++;
		state_hash[hash] = exp_realloc(state_hash[hash], sizeof(int) * state_hash[hash][0]+1);
	} else {
		state_hash[hash] = malloc(sizeof(int) * 2);
		state_hash[hash][0] = 1;
	}
	state_hash[hash][state_hash[hash][0]] = state_pool_size-1;

	if (state_pool_size % 100000 == 0)
		fprintf(stderr, "Analyzed %d states (%d moves, %d states/move), still going..\n", state_pool_size, search_level, cycle_size);

	for (i = 0; i < w*h; i++)
		if (finish_mask[i] && finish_mask[i] != state[i])
			return;
	finish_state = state_pool_size-1;
}

static void explore(int idx)
{
	int8_t *next_state = malloc(w*h);
	int8_t p, dx, dy;

	for (p = 1; p <= np; p++)
	for (dx = -1; dx <= +1; dx++)
	for (dy = -1; dy <= +1; dy++) {
		if ((!dx && !dy) || (dx && dy))
			continue;
		if (make_new(next_state, state_pool[idx], p, dx, dy))
			add_state(next_state, idx, p);
	}

	free(next_state);
}

static int geti()
{
	int rc, v;
	rc = scanf("%d", &v);
	if (rc != 1) {
		int ch = getchar();
		if ('A' <= ch && ch <= 'Z')
			return 10 + ch - 'A';
		fprintf(stderr, "Unexpected character or end of file!\n");
		exit(1);
	}
	return v;
}

static void print_state(int idx)
{
	int8_t x, y;
	printf("s%d [ label=<<TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\">", idx);
	for (y = 0; y < h; y++) {
		printf("<TR>");
		for (x = 0; x < w; x++) {
			int p = state_pool[idx][y*w + x];
			const char *col = "";
			if (block_infos[idx] && p == block_infos[idx])
				col = " BGCOLOR=\"lightblue\"";
			if (p < 10)
				printf("<TD%s>%d</TD>", col, p);
			else
				printf("<TD%s>%c</TD>", col, 'A' + p - 10);
		}
		printf("</TR>");
	}
	printf("</TABLE>> ];\n");
}

int main()
{
	int8_t *start;
	int8_t x, y;
	int i, j;

	w = geti();
	h = geti();
	np = geti();

	start = malloc(w*h);
	for (y = 0; y < h; y++)
	for (x = 0; x < w; x++)
		start[y*w + x] = geti();

	finish_mask = malloc(w*h);
	for (y = 0; y < h; y++)
	for (x = 0; x < w; x++)
		finish_mask[y*w + x] = geti();

	state_pool_size = 1;
	state_pool = malloc(sizeof(int8_t*));
	state_pool[0] = start;

	edge_infos = malloc(sizeof(int*));
	edge_infos[0] = 0;
	block_infos = malloc(sizeof(int*));
	block_infos[0] = 0;

	i = 0, j = 1;
	while (i < j && !finish_state) {
		cycle_size = j - i;
		while (i < j && !finish_state)
			explore(i++);
		j = state_pool_size;
		search_level++;
	}

	if (finish_state)
	{
		i = finish_state, j = 0;
		printf("digraph \"slidingblocks\" { rankdir=\"LR\";\n");
		while (1) {
			print_state(i);
			if (i == 0)
				break;
			printf("s%d -> s%d;\n", edge_infos[i], i);
			i = edge_infos[i], j++;
		}
		printf("};\n");
		fprintf(stderr, "Found solution with %d moves, visited %d states.\n", j, state_pool_size);
	}
	else
	{
		if (state_pool_size < 1000) {
			printf("digraph \"slidingblocks\" { rankdir=\"LR\";\n");
			for (i = 0; i < state_pool_size; i++)
				print_state(i);
			for (i = 1; i < state_pool_size; i++)
				printf("s%d -> s%d;\n", edge_infos[i], i);
			printf("};\n");
		}
		fprintf(stderr, "The %d reachable states do not contain a solution.\n", state_pool_size);
	}

	for (i = 0; i < HASH_SIZE; i++)
		if (state_hash[i])
			free(state_hash[i]);
	for (i = 0; i < state_pool_size; i++)
		free(state_pool[i]);
	free(state_pool);
	free(edge_infos);
	free(block_infos);
	free(finish_mask);

	return 0;
}

