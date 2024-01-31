// g++ -Wall -o demo01 demo01.cc && ./demo01
#define RVINTRIN_EMULATE 1

#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <algorithm>
#include "rvintrin.h"

// xorshift64
uint64_t rng()
{
	static uint64_t x = 123456789;
	x ^= x << 13;
	x ^= x >> 7;
	x ^= x << 17;
	return x;
}

typedef struct {
	int perm[64];
	uint64_t ctrl[4];
	uint64_t mask[4];
} perm64t;

uint64_t perm64_reference(perm64t &p, uint64_t v)
{
	uint64_t r = 0;
	for (int i = 0; i < 64; i++) {
		uint64_t b = (v >> p.perm[i]) & 1;
		r |= b << i;
	}
	return r;
}

void perm64_bitmanip_setup(perm64t &p)
{
	for (int i = 0; i < 4; i++) {
		p.ctrl[i] = 0;
		p.mask[i] = 0;
	}

	for (int to_idx = 0; to_idx < 64; to_idx++)
	{
		int to_nibble = to_idx / 4;
		int to_subidx = to_idx % 4;

		int frm_idx = p.perm[to_idx];
		int frm_nibble = frm_idx / 4;
		int frm_subidx = frm_idx % 4;

		p.ctrl[to_subidx] |= (uint64_t)frm_nibble << (4*to_nibble);
		p.mask[to_subidx] |= 1LL << (frm_subidx + 4*to_nibble);
	}
}

uint64_t perm64_bitmanip(perm64t &p, uint64_t v)
{
	uint64_t v0 = _rv64_xperm_n(v, p.ctrl[0]) & p.mask[0];  //  2 instructions
	uint64_t v1 = _rv64_xperm_n(v, p.ctrl[1]) & p.mask[1];  //  4 instructions
	uint64_t v2 = _rv64_xperm_n(v, p.ctrl[2]) & p.mask[2];  //  6 instructions
	uint64_t v3 = _rv64_xperm_n(v, p.ctrl[3]) & p.mask[3];  //  8 instructions

	v0 = _rv64_gorc(v0, 3) & 0x1111111111111111LL;       // 10 instructions
	v1 = _rv64_gorc(v1, 3) & 0x2222222222222222LL;       // 12 instructions
	v2 = _rv64_gorc(v2, 3) & 0x4444444444444444LL;       // 14 instructions
	v3 = _rv64_gorc(v3, 3) & 0x8888888888888888LL;       // 16 instructions

	v0 |= v1;                                            // 17 instructions
	v2 |= v3;                                            // 18 instructions
	return v0 | v2;                                      // 19 instructions
}

uint64_t perm64_bitmanip_cmix(perm64t &p, uint64_t v)
{
	uint64_t v0 = _rv64_gorc(_rv64_xperm_n(v, p.ctrl[0]) & p.mask[0], 3);   //  3 instructions
	uint64_t v1 = _rv64_gorc(_rv64_xperm_n(v, p.ctrl[1]) & p.mask[1], 3);   //  6 instructions
	uint64_t v2 = _rv64_gorc(_rv64_xperm_n(v, p.ctrl[2]) & p.mask[2], 3);   //  9 instructions
	uint64_t v3 = _rv64_gorc(_rv64_xperm_n(v, p.ctrl[3]) & p.mask[3], 3);   // 12 instructions

	v0 = _rv_cmix(0x1111111111111111LL, v0, v1);                         // 13 instructions
	v2 = _rv_cmix(0x4444444444444444LL, v2, v3);                         // 14 instructions
	return _rv_cmix(0x3333333333333333LL, v0, v2);                       // 15 instructions
}

uint64_t perm64_bitmanip_pure(perm64t &p, uint64_t v)
{
	uint64_t v0 = _rv64_xperm_n(v, p.ctrl[0]) & p.mask[0];  // +2 instructions = 2
	uint64_t v1 = _rv64_xperm_n(v, p.ctrl[1]) & p.mask[1];  // +2 instructions = 4
	uint64_t v2 = _rv64_xperm_n(v, p.ctrl[2]) & p.mask[2];  // +2 instructions = 6
	uint64_t v3 = _rv64_xperm_n(v, p.ctrl[3]) & p.mask[3];  // +2 instructions = 8

	v0 = _rv64_xperm_n(0x1111111111111110LL, v0);  // +1 instruction =  9
	v1 = _rv64_xperm_n(0x2222222222222220LL, v1);  // +1 instruction = 10
	v2 = _rv64_xperm_n(0x4444444444444440LL, v2);  // +1 instruction = 11
	v3 = _rv64_xperm_n(0x8888888888888880LL, v3);  // +1 instruction = 12

	return v0 | v1 | v2 | v3;  // +3 instructions = 15
}

int main()
{
	for (int i = 0; i < 10; i++)
		rng();

	for (int k = 0; k < 1000; k++)
	{
		perm64t p;

		for (int i = 0; i < 64; i++)
			p.perm[i] = i;

		for (int i = 0; i < 64; i++) {
			int j = rng() & 63;
			std::swap(p.perm[i], p.perm[j]);
		}

		printf("\n");
		for (int i = 63; i >= 0; i--)
			printf(" %2d", i);
		printf("\n");
		for (int i = 63; i >= 0; i--)
			printf(" %2d", p.perm[i]);
		printf("\n");

		perm64_bitmanip_setup(p);

		for (int i = 0; i < 64; i++)
		{
			uint64_t din = rng();
			uint64_t dout_reference = perm64_reference(p, din);
			uint64_t dout_bitmanip1 = perm64_bitmanip(p, din);
			uint64_t dout_bitmanip2 = perm64_bitmanip_cmix(p, din);
			uint64_t dout_bitmanip3 = perm64_bitmanip_pure(p, din);
			bool okay = (dout_reference == dout_bitmanip1) &&
					(dout_reference == dout_bitmanip2) && (dout_reference == dout_bitmanip3);
			printf("%016llx %016llx %016llx %016llx %016llx %s\n", (long long)din, (long long)dout_reference,
					(long long)dout_bitmanip1, (long long)dout_bitmanip2, (long long)dout_bitmanip3,
					okay ? "OK" : "ERROR");
			if (!okay) return 1;
		}
	}

	printf("\n");
	printf("ALL PASSED.\n");
	printf("\n");

	return 0;
}
