// g++ -Wall -o demo02 demo02.cc && ./demo02
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

uint8_t sbox[256];
uint64_t sbox_packed[32];

void sbox_pack()
{
	for (int i = 0; i < 32; i++)
		sbox_packed[i] = 0;

	for (int i = 0; i < 256; i++)
	{
		int supidx = i / 8;
		int subidx = i % 8;
		sbox_packed[supidx] |= (uint64_t)sbox[i] << (8*subidx);
	}
}

uint64_t sbox_reference(uint64_t v)
{
	uint64_t r = 0;
	for (int i = 0; i < 64; i += 8)
		r |= (uint64_t)sbox[(v >> i) & 255] << i;
	return r;
}

uint64_t lut8(uint64_t d, uint64_t s)
{
	uint64_t o = 0x1010101010101010LL;
	uint64_t p = 0xf8f8f8f8f8f8f8f8LL;
	uint64_t m = _rv64_gorc(s & p, 7);     // 2 instructions  (+2)
	s = _rv64_gorc((s & ~p) << 1, 4) + o;  // 6 instructions  (+4)
	return _rv64_xperm_n(d, s) & ~m;       // 8 instructions  (+2)
}

uint64_t sbox_bitmanip_lut8(uint64_t v)
{
	uint64_t r = 0, m = 0;
	for (int i = 0; i < 32; i++) {
		r |= lut8(sbox_packed[i], v ^ m);
		m += 0x0808080808080808LL;
	}
	return r;
}

uint64_t sbox_bitmanip_xperm(uint64_t v)
{
	uint64_t r = 0, m = 0;
	for (int i = 0; i < 32; i++) {
		r |= _rv64_xperm_b(sbox_packed[i], v ^ m);
		m += 0x0808080808080808LL;
	}
	return r;
}

// processing 4*8=32 bytes in 32*14 instructions -> 14 instructions per byte
void sbox_bitmanip_xperm4(const uint64_t din[4], uint64_t dout[4])
{
	uint64_t D0 = din[0], D1 = din[1], D2 = din[2], D3 = din[3];
	uint64_t A = 0, B = 0, C = 0, D = 0, m = 0;

	#pragma GCC unroll 32
	for (int i = 0; i < 32; i++) {
		uint64_t S = sbox_packed[i];    //  1 instruction  (+1)
		A |= _rv64_xperm_b(S, D0 ^ m);  //  4 instructions (+3)
		B |= _rv64_xperm_b(S, D1 ^ m);  //  7 instructions (+3)
		C |= _rv64_xperm_b(S, D2 ^ m);  // 10 instructions (+3)
		D |= _rv64_xperm_b(S, D3 ^ m);  // 13 instructions (+3)
		m += 0x0808080808080808LL;      // 14 instructions (+1)
	}

	dout[0] = A, dout[1] = B, dout[2] = C, dout[3] = D;
}

int main()
{
	for (int i = 0; i < 256; i++)
		sbox[i] = i;

	for (int i = 0; i < 256; i++) {
		int j = rng() & 255;
		std::swap(sbox[i], sbox[j]);
	}

	sbox_pack();

	for (int i = 0; i < 32; i++)
		printf("== %2d %016llx\n", i, (long long)sbox_packed[i]);

	for (int i = 0; i < 1000; i++)
	{
		uint64_t din = rng();
		uint64_t dout_reference = sbox_reference(din);
		uint64_t dout_bitmanip1 = sbox_bitmanip_lut8(din);
		uint64_t dout_bitmanip2 = sbox_bitmanip_xperm(din);
		bool okay = (dout_reference == dout_bitmanip1) &&
				(dout_reference == dout_bitmanip2);
		printf("%016llx %016llx %016llx %016llx %s\n", (long long)din,
				(long long)dout_reference, (long long)dout_bitmanip1,
				(long long)dout_bitmanip2, okay ? "OK" : "ERROR");
		if (!okay) return 1;
	}

	for (int i = 0; i < 1000; i++)
	{
		uint64_t din[4], dout_reference[4], dout_bitmanip[4];

		for (int k = 0; k < 4; k++) {
			din[k] = rng();
			dout_reference[k] = sbox_reference(din[k]);
		}

		sbox_bitmanip_xperm4(din, dout_bitmanip);

		for (int k = 0; k < 4; k++) {
			bool okay = dout_reference[k] == dout_bitmanip[k];
			printf("%d: %016llx %016llx %016llx %s\n", k,
					(long long)din[k], (long long)dout_reference[k],
					(long long)dout_bitmanip[k], okay ? "OK" : "ERROR");
			if (!okay) return 1;
		}
	}

	printf("\n");
	printf("ALL PASSED.\n");
	printf("\n");

	return 0;
}
