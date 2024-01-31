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
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>

static bool verbose = false;

uint64_t xorshift64()
{
	static uint64_t x64 = 88172645463325252ull;
	x64 ^= x64 << 13;
	x64 ^= x64 >> 7;
	x64 ^= x64 << 17;
	return x64;
}

void print_bin(const char *label, uint64_t value, int width)
{
	printf("  %s 0b", label);
	for (int i = 0; i < 64; i++) {
		if (width++ >= 64)
			printf("%d", int(value >> 63));
		value <<= 1;
	}
	printf("\n");
}

uint64_t lrotc_zero(int width, int cycles)
{
	uint64_t value = 0;
	while (cycles--) {
		if (value >> (width-1))
			value = (value ^ (1 << (width-1))) << 1;
		else
			value = (value << 1) | 1;
	}
	return value;
}

void decoder(uint64_t mask, uint32_t ctrl[6])
{
	int ppc[63];

	if (verbose) {
		printf("decoder\n");
		print_bin("mask", mask, 64);
	}

	ppc[0] = (mask & 1);
	for (int i = 1; i < 63; i++) {
		mask >>= 1;
		ppc[i] = ppc[i-1] + (mask & 1);
	}

	for (int i = 0; i < 6; i++) {
		int k = 1 << i;
		ctrl[i] = 0;
		for (int j = 0; j < 32/k; j++) {
			uint32_t temp = lrotc_zero(k, ppc[2*j*k + k - 1]);
			ctrl[i] |= temp << (j*k);
		}
	}

	if (verbose) {
		print_bin("ctrl[0]", ctrl[0], 32);
		print_bin("ctrl[1]", ctrl[1], 32);
		print_bin("ctrl[2]", ctrl[2], 32);
		print_bin("ctrl[3]", ctrl[3], 32);
		print_bin("ctrl[4]", ctrl[4], 32);
		print_bin("ctrl[5]", ctrl[5], 32);
	}
}

uint64_t butterfly(uint64_t value, uint32_t ctrl, int stage)
{
	uint64_t result = 0;

	for (int i = 0; i < (32 >> stage); i++)
	for (int j = 0; j < (1 << stage); j++)
	{
		int idx1 = i * (2 << stage) + j;
		int idx2 = idx1 + (1 << stage);

		if (ctrl & 1) {
			if (value & (uint64_t(1) << idx1)) result |= (uint64_t(1) << idx1);
			if (value & (uint64_t(1) << idx2)) result |= (uint64_t(1) << idx2);
		} else {
			if (value & (uint64_t(1) << idx1)) result |= (uint64_t(1) << idx2);
			if (value & (uint64_t(1) << idx2)) result |= (uint64_t(1) << idx1);
		}

		ctrl >>= 1;
	}

	return result;
}

uint64_t pext_butterfly(uint64_t value, uint64_t mask)
{
	uint32_t ctrl[6];
	decoder(mask, ctrl);

	uint64_t v0 = value & mask;
	uint64_t v1 = butterfly(v0, ctrl[0], 0);
	uint64_t v2 = butterfly(v1, ctrl[1], 1);
	uint64_t v3 = butterfly(v2, ctrl[2], 2);
	uint64_t v4 = butterfly(v3, ctrl[3], 3);
	uint64_t v5 = butterfly(v4, ctrl[4], 4);
	uint64_t v6 = butterfly(v5, ctrl[5], 5);

	if (verbose) {
		printf("butterfly\n");
		print_bin("v0", v0, 64);
		print_bin("v1", v1, 64);
		print_bin("v2", v2, 64);
		print_bin("v3", v3, 64);
		print_bin("v4", v4, 64);
		print_bin("v5", v5, 64);
		print_bin("v6", v6, 64);
	}

	return v6;
}

uint64_t pdep_butterfly(uint64_t value, uint64_t mask)
{
	uint32_t ctrl[6];
	decoder(mask, ctrl);

	uint64_t v0 = value;
	uint64_t v1 = butterfly(v0, ctrl[5], 5);
	uint64_t v2 = butterfly(v1, ctrl[4], 4);
	uint64_t v3 = butterfly(v2, ctrl[3], 3);
	uint64_t v4 = butterfly(v3, ctrl[2], 2);
	uint64_t v5 = butterfly(v4, ctrl[1], 1);
	uint64_t v6 = butterfly(v5, ctrl[0], 0);

	if (verbose) {
		printf("butterfly\n");
		print_bin("v0", v0, 64);
		print_bin("v1", v1, 64);
		print_bin("v2", v2, 64);
		print_bin("v3", v3, 64);
		print_bin("v4", v4, 64);
		print_bin("v5", v5, 64);
		print_bin("v6", v6, 64);
	}

	return v6 & mask;
}

uint64_t pext_naive(uint64_t value, uint64_t mask)
{
	int t = 0;
	uint64_t c = 0;
	for (int i = 0; i < 64; i++) {
		if (mask & 1) {
			c |= (value & 1) << (t++);
		}
		value >>= 1;
		mask >>= 1;
	}
	return c;
}

uint64_t pdep_naive(uint64_t value, uint64_t mask)
{
	uint64_t c = 0;
	for (int i = 0; i < 64; i++) {
		if (mask & 1) {
			c |= (value & 1) << i;
			value >>= 1;
		}
		mask >>= 1;
	}
	return c;
}

uint64_t grev(uint64_t x, uint64_t k)
{
	if (k & 1 ) x = ((x & 0x5555555555555555ull) << 1 ) | ((x & 0xAAAAAAAAAAAAAAAAull) >> 1 );
	if (k & 2 ) x = ((x & 0x3333333333333333ull) << 2 ) | ((x & 0xCCCCCCCCCCCCCCCCull) >> 2 );
	if (k & 4 ) x = ((x & 0x0F0F0F0F0F0F0F0Full) << 4 ) | ((x & 0xF0F0F0F0F0F0F0F0ull) >> 4 );
	if (k & 8 ) x = ((x & 0x00FF00FF00FF00FFull) << 8 ) | ((x & 0xFF00FF00FF00FF00ull) >> 8 );
	if (k & 16) x = ((x & 0x0000FFFF0000FFFFull) << 16) | ((x & 0xFFFF0000FFFF0000ull) >> 16);
	if (k & 32) x = ((x & 0x00000000FFFFFFFFull) << 32) | ((x & 0xFFFFFFFF00000000ull) >> 32);
	return x;
}

void test(uint64_t value, uint64_t mask)
{
	printf("\n");

	printf("input\n");
	print_bin("value", value, 64);
	print_bin("mask ", mask, 64);

	uint64_t pext_result_naive = pext_naive(value, mask);
	uint64_t pext_result_btfly = pext_butterfly(value, mask);

	printf("output_pext\n");
	print_bin("naive", pext_result_naive, 64);
	print_bin("btfly", pext_result_btfly, 64);

	assert(pext_result_naive == pext_result_btfly);

	uint64_t pdep_result_naive = pdep_naive(value, mask);
	uint64_t pdep_result_btfly = pdep_butterfly(value, mask);

	printf("output_pdep\n");
	print_bin("naive", pdep_result_naive, 64);
	print_bin("btfly", pdep_result_btfly, 64);

	assert(pdep_result_naive == pdep_result_btfly);

	uint64_t grev_result = grev(value, mask);

	printf("output_grev\n");
	print_bin("btfly", grev_result, 64);

	printf("test(64'h%016llx, 64'h%016llx, 64'h%016llx, 64'h%016llx, 64'h%016llx);\n",
			(long long)value, (long long)mask, (long long)pext_result_naive,
			(long long)pdep_result_naive, (long long)grev_result);
}

int main()
{
	for (int i = 0; i < 1000; i++) {
		uint64_t value = xorshift64();
		uint64_t mask = xorshift64();
		test(value, mask);
	}

	printf("\n");
	printf("OK.\n");
	return 0;
}

