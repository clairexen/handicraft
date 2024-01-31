/*
 *  Copyright (C) 2018  Clifford Wolf <clifford@clifford.at>
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

#include <stdint.h>
#include <stdio.h>

#define PRINT_TRACE

static const uint64_t keccak_round_constants[24] = {
	0x0000000000000001LL,
	0x0000000000008082LL,
	0x800000000000808ALL,
	0x8000000080008000LL,
	0x000000000000808BLL,
	0x0000000080000001LL,
	0x8000000080008081LL,
	0x8000000000008009LL,
	0x000000000000008ALL,
	0x0000000000000088LL,
	0x0000000080008009LL,
	0x000000008000000ALL,
	0x000000008000808BLL,
	0x800000000000008BLL,
	0x8000000000008089LL,
	0x8000000000008003LL,
	0x8000000000008002LL,
	0x8000000000000080LL,
	0x000000000000800ALL,
	0x800000008000000ALL,
	0x8000000080008081LL,
	0x8000000000008080LL,
	0x0000000080000001LL,
	0x8000000080008008LL
};

void print_state_1d(uint64_t data[25])
{
	for (int y = 0; y < 5; y++)
	{
		printf("%016llX %016llX %016llX %016llX %016llX\n",
			(long long)data[5*y + 0], (long long)data[5*y + 1], (long long)data[5*y + 2],
			(long long)data[5*y + 3], (long long)data[5*y + 4]);
	}
}

void print_state_2d(uint64_t data[5][5])
{
	for (int y = 0; y < 5; y++)
	{
		printf("%016llX %016llX %016llX %016llX %016llX\n",
			(long long)data[0][y], (long long)data[1][y], (long long)data[2][y],
			(long long)data[3][y], (long long)data[4][y]);
	}
}

void keccak_p_reference(uint64_t state[25], uint64_t round_constant)
{
	// Input

	uint64_t A[5][5];

	for (int x = 0; x < 5; x++)
	for (int y = 0; y < 5; y++)
	{
		A[x][y] = state[5*y + x];
	}

	// Theta Step

	uint64_t C[5];

	for (int x = 0; x < 5; x++)
	{
		C[x] = 0;
		for (int y = 0; y < 5; y++)
			C[x] ^= A[x][y];
	}

	for (int x = 0; x < 5; x++)
	{
		int x_left = (x+4) % 5, x_right = (x+1) % 5;
		uint64_t D = C[x_left] ^ (C[x_right] << 1) ^ (C[x_right] >> 63);
		for (int y = 0; y < 5; y++)
			A[x][y] ^= D;
	}

#ifdef PRINT_TRACE
	printf("After theta:\n");
	print_state_2d(A);
#endif

	// Rho Step

	for (int x = 0; x < 5; x++)
	for (int y = 0; y < 5; y++)
	{
		static const int rot_table[5][5] = {
			{ 0, 36,  3, 41, 18}, // x=0
			{ 1, 44, 10, 45,  2}, // x=1
			{62,  6, 43, 15, 61}, // x=2
			{28, 55, 25, 21, 56}, // x=3
			{27, 20, 39,  8, 14}  // x=4
		};
		
		int r = rot_table[x][y];
		int r_inverse = (64-r) & 63;
		A[x][y] = (A[x][y] << r) | (A[x][y] >> r_inverse);
	}

#ifdef PRINT_TRACE
	printf("After rho:\n");
	print_state_2d(A);
#endif

	// Pi Step

	uint64_t B[5][5];

	for (int x = 0; x < 5; x++)
	for (int y = 0; y < 5; y++)
	{
		B[y][(2*x + 3*y) % 5] = A[x][y];
	}

#ifdef PRINT_TRACE
	printf("After pi:\n");
	print_state_2d(B);
#endif

	// Chi Step

	for (int x = 0; x < 5; x++)
	for (int y = 0; y < 5; y++)
	{
		int x_right_1 = (x+1) % 5, x_right_2 = (x+2) % 5;
		A[x][y] = B[x][y] ^ (~B[x_right_1][y] & B[x_right_2][y]);
	}

#ifdef PRINT_TRACE
	printf("After chi:\n");
	print_state_2d(A);
#endif

	// Iota Step

	A[0][0] ^= round_constant;

#ifdef PRINT_TRACE
	printf("After iota:\n");
	print_state_2d(A);
#endif

	// Output

	for (int x = 0; x < 5; x++)
	for (int y = 0; y < 5; y++)
	{
		state[5*y + x] = A[x][y];
	}
}

void keccak_f_reference(uint64_t state[25])
{
#ifdef PRINT_TRACE
	printf("Input:\n");
	print_state_1d(state);
#endif

	for (int i = 0; i < 24; i++) {
#ifdef PRINT_TRACE
		printf("\n--- Round %d ---\n\n", i);
#endif
		keccak_p_reference(state, keccak_round_constants[i]);
	}

#ifdef PRINT_TRACE
	printf("\nOutput:\n");
	print_state_1d(state);
#endif
}

int main()
{
	uint64_t state[25] = { /* zeros */ };
	keccak_f_reference(state);
	return 0;
}
