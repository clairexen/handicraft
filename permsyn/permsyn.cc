/*
 *  Copyright (C) 2017  Clifford Wolf <clifford@clifford.at>
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
#include <assert.h>
#include <vector>

#define PROFILE_N 500

#define OP_INIT 0
#define OP_ROT  1
#define OP_GREV 2
#define OP_BEXT 3
#define OP_BDEP 4
#define OP_AND  5
#define OP_OR   6

struct node_t
{
	int8_t value[32];
	uint32_t rbits = 0;
	uint32_t arg = 0;
	int8_t op = 0;
	int8_t src1 = 0;
	int8_t src2 = 0;
};

std::vector<node_t> stack;
std::vector<node_t> best_stack;
int stack_frame_ptr = 0;
uint32_t solved_bits = ~0;
int best_score = 0;

bool use_grev = true;
bool use_bext_bdep = true;
bool gen_c = false;
bool profile = false;
bool simple = false;
bool gendata = false;

void score_stack()
{
	int len = stack.size() - stack_frame_ptr;
	bool need_mask = false;
	uint32_t rbits = 0;
	int score = -len*49;

	if (len == 0)
		return;

	for (int i = 0; i < 32; i++)
		if (stack.back().value[i] == i) {
			uint32_t mask = 1 << i;
			rbits |= mask;
			if (mask & ~solved_bits)
				score += 150;
		} else
		if (stack.back().value[i] < 32)
			need_mask = true;

	if (need_mask)
		score -= 50;

	if (score > best_score)
	{
		best_stack.resize(stack.size() - stack_frame_ptr);

		for (int i = 0; i < len; i++)
			best_stack[i] = stack[stack_frame_ptr+i];

		if (need_mask)
		{
			node_t node;

			node.op = OP_AND;
			node.src1 = stack.size()-1;
			node.arg = rbits;

			for (int i = 0; i < 32; i++)
				node.value[i] = (rbits & (1 << i)) ? stack.back().value[i] : 32;

			best_stack.push_back(node);
		}

		best_stack.back().rbits = rbits;
		best_score = score;
	}
}

void score_node(int n)
{
	bool need_mask = false;
	uint32_t rbits = 0;
	int score = -50;

	if (stack[n].rbits)
		return;

	for (int i = 0; i < 32; i++)
		if (stack[n].value[i] == i) {
			uint32_t mask = 1 << i;
			rbits |= mask;
			if (mask & ~solved_bits)
				score += 150;
		} else
		if (stack[n].value[i] < 32)
			need_mask = true;
	
	if (!need_mask) {
		stack[n].rbits = rbits;
		solved_bits |= rbits;
		return;
	}

	if (score > best_score)
	{
		node_t node;

		node.op = OP_AND;
		node.src1 = n;
		node.arg = rbits;
		node.rbits = rbits;

		for (int i = 0; i < 32; i++)
			node.value[i] = (rbits & (1 << i)) ? stack[n].value[i] : 32;

		best_stack.clear();
		best_stack.push_back(node);
		best_score = score;
	}
}

void print_node(int n)
{
	if (profile || gendata)
		return;

	printf("// %c %2d: ", stack[n].rbits ? '*' : ' ', n);
	switch (stack[n].op) {
		case OP_INIT: printf("initial           "); break;
		case OP_ROT:  printf("rot  %2d %2d        ", stack[n].src1, stack[n].arg); break;
		case OP_GREV: printf("grev %2d 0x%02x      ", stack[n].src1, stack[n].arg); break;
		case OP_BEXT: printf("bext %2d 0x%08x", stack[n].src1, stack[n].arg); break;
		case OP_BDEP: printf("bdep %2d 0x%08x", stack[n].src1, stack[n].arg); break;
		case OP_AND:  printf("and  %2d 0x%08x", stack[n].src1, stack[n].arg); break;
		case OP_OR:   printf("or   %2d %2d        ", stack[n].src1, stack[n].src2); break;
		default: abort();
	}
	printf(" -> ");
	for (int i = 31; i >= 0; i--) {
		putchar((stack[n].rbits & (1 << i)) != 0 ? '[' : ' ');
		if (stack[n].value[i] == 32)
			printf("--");
		else
			printf("%2d", stack[n].value[i]);
		putchar((stack[n].rbits & (1 << i)) != 0 ? ']' : ' ');
	}
	printf("\n");
}

void push_rot(int8_t src, int arg)
{
	node_t node;

	node.op = OP_ROT;
	node.src1 = src;
	node.arg = arg;

	for (int i = 0; i < 32; i++)
		node.value[i] = stack[src].value[(i+arg) % 32];

	stack.push_back(node);
	score_stack();
}

void push_grev(int8_t src, int arg)
{
	node_t node;

	node.op = OP_GREV;
	node.src1 = src;
	node.arg = arg;

	for (int i = 0; i < 32; i++)
		node.value[i] = stack[src].value[i];

	for (int i = 0; i < 5; i++)
	{
		int offset = 1 << i;

		if ((arg & offset) == 0)
			continue;

		for (int k = 0; k < 16; k++) {
			int idx = (2*offset * (k / offset)) + (k % offset);
			std::swap(node.value[idx], node.value[idx+offset]);
		}
	}

	stack.push_back(node);
	score_stack();
}

void best_bext_bdep(int n)
{
	if (!use_bext_bdep)
		return;

	// --- BEGIN: Longest Increasing Subsequence Search ---
	// Code from https://en.wikipedia.org/wiki/Longest_increasing_subsequence with modifications

	int8_t X[32];
	int8_t P[32];
	int8_t S[32];
	int8_t M[33];

	int N = 0;
	int L = 0;

	M[0] = 0;

	for (int v : stack[n].value)
		if (v < 32 && (solved_bits & (1 << v)) == 0)
			X[N++] = v;

	for (int i = 0; i < N; i++)
	{
		// Binary search for the largest positive j <= L
		// such that X[M[j]] < X[i]
		int lo = 1;
		int hi = L;

		while (lo <= hi)
		{
			int mid = (lo+hi)/2;

			if (X[M[mid]] < X[i])
				lo = mid+1;
			else
				hi = mid-1;
		}

		// After searching, lo is 1 greater than the
		// length of the longest prefix of X[i]
		int newL = lo;

		// The predecessor of X[i] is the last index of 
		// the subsequence of length newL-1
		P[i] = M[newL-1];
		M[newL] = i;

		// If we found a subsequence longer than any we've
		// found yet, update L
		if (newL > L)
			L = newL;
	}

	// Reconstruct the longest increasing subsequence
	for (int k = M[L], i = L-1; i >= 0; i--) {
		S[i] = X[k];
		k = P[k];
	}
	// --- END: Longest Increasing Subsequence Search ---

	if (L < 2)
		return;

	uint32_t ext_mask = 0;
	uint32_t dep_mask = 0;

	for (int i = 0, k = 0; i < 32 && k < L; i++)
		if (stack[n].value[i] == S[k]) {
			ext_mask |= 1 << i;
			dep_mask |= 1 << S[k++];
		}
	
	node_t ext_node;
	ext_node.op = OP_BEXT;
	ext_node.src1 = n;
	ext_node.arg = ext_mask;

	node_t dep_node;
	dep_node.op = OP_BDEP;
	dep_node.src1 = stack.size();
	dep_node.arg = dep_mask;

	for (int i = 0; i < 32; i++)
	{
		ext_node.value[i] = i < L ? S[i] : 32;
		dep_node.value[i] = dep_mask & (1 << i) ? i : 32;
	}

	stack.push_back(ext_node);
	stack.push_back(dep_node);
	score_stack();
	stack.pop_back();
	stack.pop_back();
}

void search(int n)
{
	if (stack[n].op == OP_BEXT || stack[n].op == OP_BDEP || stack[n].op == OP_AND)
		return;

	score_node(n);
	best_bext_bdep(stack.size()-1);

	if (simple)
	{
		if (stack[n].op != OP_GREV && use_grev)
		{
			for (int i = 1; i < 32; i++)
			{
				push_grev(n, i);
				best_bext_bdep(stack.size()-1);
				for (int j = 1; j < 32; j++) {
					push_rot(stack.size()-1, j);
					best_bext_bdep(stack.size()-1);
					stack.pop_back();
				}
				stack.pop_back();
			}
		} else
		if (stack[n].op != OP_ROT)
		{
			for (int i = 1; i < 32; i++)
			{
				push_rot(n, i);
				best_bext_bdep(stack.size()-1);
				stack.pop_back();
			}
		}
	}
	else
	{
		if (stack[n].op != OP_GREV && use_grev)
		{
			for (int i = 1; i < 32; i++)
			{
				push_grev(n, i);
				best_bext_bdep(stack.size()-1);
				for (int j = 1; j < 32; j++) {
					push_rot(stack.size()-1, j);
					best_bext_bdep(stack.size()-1);
					for (int k = 1; k < 32; k++) {
						push_grev(n, k);
						best_bext_bdep(stack.size()-1);
						stack.pop_back();
					}
					stack.pop_back();
				}
				stack.pop_back();
			}
		}

		if (stack[n].op != OP_ROT)
		{
			for (int i = 1; i < 32; i++)
			{
				push_rot(n, i);
				best_bext_bdep(stack.size()-1);
				if (use_grev) {
					for (int j = 1; j < 32; j++) {
						push_grev(stack.size()-1, j);
						best_bext_bdep(stack.size()-1);
						for (int k = 1; k < 32; k++) {
							push_rot(n, k);
							best_bext_bdep(stack.size()-1);
							stack.pop_back();
						}
						stack.pop_back();
					}
				}
				stack.pop_back();
			}
		}
	}
}

void test(char *p)
{
	if (*p == 0) {
		auto &v = stack.back().value;
		for (int i = 31; i >= 0; i--)
			putchar(v[i] < 10 ? '0' + v[i] : 'a' + v[i] - 10);
		putchar('\n');
		return;
	}

	if (*p == 'g') {
		for (int i = 0; i < 32; i++) {
			push_grev(stack.size()-1, i);
			test(p+1);
			stack.pop_back();
		}
		return;
	}

	if (*p == 'r') {
		for (int i = 0; i < 32; i++) {
			push_rot(stack.size()-1, i);
			test(p+1);
			stack.pop_back();
		}
		return;
	}

	abort();
}

int main(int argc, char **argv)
{
	node_t root;

	if (argc < 2) {
usage:
		fprintf(stderr, "Usage: %s [-p] [-d] [-s] [-c] [-g] [-b] [0-9a-v-]{32}\n", argv[0]);
		fprintf(stderr, "       %s -t [gr]+ ...\n", argv[0]);
		return 1;
	}

	while (argc > 2)
	{
		if (!strcmp(argv[1], "-p")) {
			profile = true;
		} else
		if (!strcmp(argv[1], "-d")) {
			gendata = true;
		} else
		if (!strcmp(argv[1], "-s")) {
			simple = true;
		} else
		if (!strcmp(argv[1], "-c")) {
			gen_c = true;
		} else
		if (!strcmp(argv[1], "-g")) {
			use_grev = false;
		} else
		if (!strcmp(argv[1], "-b")) {
			use_bext_bdep = false;
		} else
		if (!strcmp(argv[1], "-t")) {
			for (int i = 0; i < 32; i++)
				root.value[i] = i;
			stack.push_back(root);
			for (int i = 2; i < argc; i++)
				test(argv[i]);
			return 1;
		} else
			goto usage;

		argv[1] = argv[0];
		argv++, argc--;
	}

	for (int i = 0; i < 32; i++)
	{
		if (argv[1][i] == 0)
			goto usage;

		if (argv[1][i] == '-') {
			root.value[31-i] = 32;
			continue;
		}

		if ('0' <= argv[1][i] && argv[1][i] <= '9') {
			root.value[31-i] = argv[1][i] - '0';
			continue;
		}

		if ('a' <= argv[1][i] && argv[1][i] <= 'v') {
			root.value[31-i] = argv[1][i] - 'a' + 10;
			continue;
		}

		goto usage;
	}

	int profile_iter = 0;

	if (profile) {
		printf("%3d%% ", 0);
		fflush(stdout);
	}

reset:
	for (int i = 0; i < 32; i++)
		if (root.value[i] < 32)
			solved_bits &= ~(1 << root.value[i]);

	stack.push_back(root);
	score_node(0);
	print_node(0);
	stack_frame_ptr = stack.size();

	while (~solved_bits)
	{
		best_score = 0;

		if (simple)
			search(0);
		else
			for (int i = 0; i < stack.size(); i++)
				search(i);

		assert(best_score > 0);
		assert(stack.size() == stack_frame_ptr);

		for (auto &node : best_stack) {
			stack.push_back(node);
			print_node(stack.size()-1);
		}

		solved_bits |= stack.back().rbits;
		stack_frame_ptr = stack.size();
	}

	if (profile)
	{
		putchar('.');

		if (++profile_iter == PROFILE_N) {
			putchar('\n');
			return 0;
		}

		if (profile_iter % 50 == 0)
			printf("\n%3d%% ", 100*profile_iter / PROFILE_N);

		fflush(stdout);

		stack.clear();
		goto reset;
	}

	std::vector<int> parts;

	for (int i = 0; i < stack.size(); i++)
		if (stack[i].rbits)
			parts.push_back(i);
	
	if (parts.size() == 1) {
		assert(parts.back() == stack.size()-1);
	}

	while (parts.size() > 1)
	{
		std::vector<int> new_parts;

		for (int i = 0; i < parts.size(); i += 2)
		{
			if (i == parts.size()-1) {
				new_parts.push_back(parts[i]);
				continue;
			}

			int n = parts[i], m = parts[i+1];

			node_t node;

			node.op = OP_OR;
			node.src1 = n;
			node.src2 = m;
			node.rbits = stack[n].rbits | stack[m].rbits;

			for (int i = 0; i < 32; i++)
				node.value[i] = (node.rbits & (1 << i)) ? i : 32;

			new_parts.push_back(stack.size());
			stack.push_back(node);
			print_node(stack.size()-1);
		}

		parts.swap(new_parts);
	}

	if (gendata)
		printf("%d\n", int(stack.size()-1));

	if (gen_c)
	{
		for (int i = 1; i < stack.size(); i++)
		{
			auto &node = stack[i];

			switch (node.op) {
				case OP_ROT:  printf("uint32_t v%d = rot(v%d, %d);\n", i, node.src1, node.arg); break;
				case OP_GREV: printf("uint32_t v%d = grev(v%d, 0x%02x);\n", i, node.src1, node.arg); break;
				case OP_BEXT: printf("uint32_t v%d = bext(v%d, 0x%08x);\n", i, node.src1, node.arg); break;
				case OP_BDEP: printf("uint32_t v%d = bdep(v%d, 0x%08x);\n", i, node.src1, node.arg); break;
				case OP_AND:  printf("uint32_t v%d = and(v%d, 0x%08x);\n", i, node.src1, node.arg); break;
				case OP_OR:   printf("uint32_t v%d = or(v%d, v%d);\n", i, node.src1, node.src2); break;
				default: abort();
			}
		}

		printf("return v%d;\n", int(stack.size()-1));
	}

	return 0;
}
