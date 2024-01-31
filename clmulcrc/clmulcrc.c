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
 */

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <zlib.h>

#if !(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#  error "This simple demo only works on little endian machines."
#endif

uint64_t clmul(uint64_t rs1, uint64_t rs2)
{
	uint64_t x = 0;
	for (int i = 0; i < 64; i++)
		if ((rs2 >> i) & 1)
			x ^= rs1 << i;
	return x;
}

uint64_t clmulh(uint64_t rs1, uint64_t rs2)
{
	uint64_t x = 0;
	for (int i = 1; i < 64; i++)
		if ((rs2 >> i) & 1)
			x ^= rs1 >> (64 - i);
	return x;
}

uint32_t clmul_32(uint32_t rs1, uint32_t rs2)
{
	uint32_t x = 0;
	for (int i = 0; i < 32; i++)
		if ((rs2 >> i) & 1)
			x ^= rs1 << i;
	return x;
}

uint32_t clmulh_32(uint32_t rs1, uint32_t rs2)
{
	uint32_t x = 0;
	for (int i = 1; i < 32; i++)
		if ((rs2 >> i) & 1)
			x ^= rs1 >> (32 - i);
	return x;
}

uint32_t clmulhx_32(uint32_t rs1, uint32_t rs2)
{
	return clmulh_32(rs1, rs2) ^ rs1;
}

uint64_t bswap(uint64_t v)
{
	v = ((v & 0x00000000ffffffffLL) << 32) | ((v & 0xffffffff00000000LL) >> 32);
	v = ((v & 0x0000ffff0000ffffLL) << 16) | ((v & 0xffff0000ffff0000LL) >> 16);
	v = ((v & 0x00ff00ff00ff00ffLL) <<  8) | ((v & 0xff00ff00ff00ff00LL) >>  8);
	return v;
}

uint32_t bswap_32(uint32_t v)
{
	v = ((v & 0x0000ffff) << 16) | ((v & 0xffff0000) >> 16);
	v = ((v & 0x00ff00ff) <<  8) | ((v & 0xff00ff00) >>  8);
	return v;
}

uint64_t brev(uint64_t v)
{
	v = ((v & 0x00000000ffffffffLL) << 32) | ((v & 0xffffffff00000000LL) >> 32);
	v = ((v & 0x0000ffff0000ffffLL) << 16) | ((v & 0xffff0000ffff0000LL) >> 16);
	v = ((v & 0x00ff00ff00ff00ffLL) <<  8) | ((v & 0xff00ff00ff00ff00LL) >>  8);
	v = ((v & 0x0f0f0f0f0f0f0f0fLL) <<  4) | ((v & 0xf0f0f0f0f0f0f0f0LL) >>  4);
	v = ((v & 0x3333333333333333LL) <<  2) | ((v & 0xccccccccccccccccLL) >>  2);
	v = ((v & 0x5555555555555555LL) <<  1) | ((v & 0xaaaaaaaaaaaaaaaaLL) >>  1);
	return v;
}

uint64_t brev_w(uint64_t v)
{
	v = ((v & 0x0000ffff0000ffffLL) << 16) | ((v & 0xffff0000ffff0000LL) >> 16);
	v = ((v & 0x00ff00ff00ff00ffLL) <<  8) | ((v & 0xff00ff00ff00ff00LL) >>  8);
	v = ((v & 0x0f0f0f0f0f0f0f0fLL) <<  4) | ((v & 0xf0f0f0f0f0f0f0f0LL) >>  4);
	v = ((v & 0x3333333333333333LL) <<  2) | ((v & 0xccccccccccccccccLL) >>  2);
	v = ((v & 0x5555555555555555LL) <<  1) | ((v & 0xaaaaaaaaaaaaaaaaLL) >>  1);
	return v;
}

// -----------------------------------------------------------------------

// CRC-32Q (Used for aeronautical data. Recognised by the ICAO.)
// http://reveng.sourceforge.net/crc-catalogue/17plus.htm#crc.cat.crc-32q
uint32_t crc32q_simple(const uint8_t *data, int length)
{
	uint32_t crc = 0;
	for (int i = 0; i < length; i++) {
		uint8_t byte = data[i];
		for (int j = 0; j < 8; j++) {
			if ((crc ^ (byte << 24)) & 0x80000000)
				crc = (crc << 1) ^ 0x814141ab;
			else
				crc = crc << 1;
			byte = byte << 1;
		}
	}
	return crc;
}

uint32_t crc32q_clmul_simple(const uint32_t *data, int length)
{
	uint32_t P  = 0x814141AB;
	uint32_t mu = 0xFEFF7F62;
	uint32_t crc = 0;

	for (int i = 0; i < length; i++) {
		crc ^= bswap_32(data[i]);
		crc = clmulhx_32(crc, mu);
		crc = clmul_32(crc, P);
	}

	return crc;
}

uint32_t crc32q_clmul(const uint64_t *p, int len)
{
	uint64_t P  = 0x1814141ABLL;
	uint64_t k1 =  0xA1FA6BECLL;
	uint64_t k2 =  0x9BE9878FLL;
	uint64_t k3 =  0xB1EFC5F6LL;
	uint64_t mu = 0x1FEFF7F62LL;

	uint64_t a0, a1, a2, t1, t2;

	assert(len >= 2);
	a0 = bswap(p[0]);
	a1 = bswap(p[1]);

	// Main loop: Reduce to 2x 64 bits

	for (const uint64_t *t0 = p+2; t0 != p+len; t0++)
	{
		a2 = bswap(*t0);
		t1 = clmulh(a0, k1);
		t2 = clmul(a0, k1);
		a0 = a1 ^ t1;
		a1 = a2 ^ t2;
	}

	// Reduce to 64 bit, add 32 bit zero padding

	t1 = clmulh(a0, k2);
	t2 = clmul(a0, k2);

	a0 = (a1 >> 32) ^ t1;
	a1 = (a1 << 32) ^ t2;

	t2 = clmul(a0, k3);
	a1 = a1 ^ t2;

	// Barrett Reduction

	t1 = clmul(a1 >> 32, mu);
	t2 = clmul(t1 >> 32, P);
	a0 = a1 ^ t2;

	return a0;
}

uint32_t crc32q_clmul_u32(const uint32_t *p, int len)
{
	uint32_t P  = 0x814141AB;
	uint32_t k3 = 0xB1EFC5F6;
	uint32_t mu = 0xFEFF7F62;

	uint32_t a0, a1, a2, t1, t2;

	assert(len >= 2);
	a0 = bswap_32(p[0]);
	a1 = bswap_32(p[1]);

	// Main loop: Reduce to 64 bits

	for (const uint32_t *t0 = p+2; t0 != p+len; t0++)
	{
		a2 = bswap_32(*t0);
		t1 = clmulh_32(a0, k3);
		t2 = clmul_32(a0, k3);
		a0 = a1 ^ t1;
		a1 = a2 ^ t2;
	}

	// add 32 bit zero padding

	t1 = clmulh_32(a0, k3);
	a2 = clmul_32(a0, k3);
	a0 = a1 ^ t1;

	// Barrett Reduction

	t1 = clmulhx_32(a0, mu);
	t2 = clmul_32(t1, P);
	a0 = a2 ^ t2;

	return a0;
}

// -----------------------------------------------------------------------

// This is equivalent to the CRC-32 used by zlib
// http://reveng.sourceforge.net/crc-catalogue/17plus.htm#crc.cat.crc-32
uint32_t crc32_simple(const uint8_t *data, int length)
{
	uint32_t crc = 0xFFFFFFFF;
	for (int i = 0; i < length; i++) {
		uint8_t byte = data[i];
		for (int j = 0; j < 8; j++) {
			if ((crc ^ byte) & 1)
				crc = (crc >> 1) ^ 0xEDB88320;
			else
				crc = crc >> 1;
			byte = byte >> 1;
		}
	}
	return ~crc;
}

uint32_t crc32_clmul(const uint64_t *p, int len)
{
	uint64_t P  = 0x104C11DB7LL;
	uint64_t k1 =  0xE8A45605LL;
	uint64_t k2 =  0xF200AA66LL;
	uint64_t k3 =  0x490D678DLL;
	uint64_t mu = 0x104D101DFLL;

	uint64_t a0, a1, a2, t1, t2;

	assert(len >= 2);
	a0 = brev(p[0] ^ 0xFFFFFFFFLL);
	a1 = brev(p[1]);

	// Main loop: Reduce to 2x 64 bits

	for (const uint64_t *t0 = p+2; t0 != p+len; t0++)
	{
		a2 = brev(*t0);
		t1 = clmulh(a0, k1);
		t2 = clmul(a0, k1);
		a0 = a1 ^ t1;
		a1 = a2 ^ t2;
	}

	// Reduce to 64 bit, add 32 bit zero padding

	t1 = clmulh(a0, k2);
	t2 = clmul(a0, k2);

	a0 = (a1 >> 32) ^ t1;
	a1 = (a1 << 32) ^ t2;

	t2 = clmul(a0, k3);
	a1 = a1 ^ t2;

	// Barrett Reduction

	t1 = clmul(a1 >> 32, mu);
	t2 = clmul(t1 >> 32, P);
	a0 = a1 ^ t2;

	return brev_w(~a0);
}

uint32_t crc32_insn_b(uint32_t x)
{
	for (int i = 0; i < 8; i++)
		x = (x >> 1) ^ (0xEDB88320 & ~((x&1)-1));
	return x;
}

uint32_t crc32_insn_h(uint32_t x)
{
	for (int i = 0; i < 16; i++)
		x = (x >> 1) ^ (0xEDB88320 & ~((x&1)-1));
	return x;
}

uint32_t crc32_insn_w(uint32_t x)
{
	for (int i = 0; i < 32; i++)
		x = (x >> 1) ^ (0xEDB88320 & ~((x&1)-1));
	return x;
}

uint32_t crc32_insn_w_clmul(uint32_t x)
{
	uint32_t P  = 0x04C11DB7;
	uint32_t mu = 0x04D101DF;
	uint32_t a0, a1;

	a0 = brev_w(x);
	a1 = clmulh_32(a0, mu);
	a1 = a1 ^ a0;
	a0 = clmul_32(a1, P);
	a0 = brev_w(a0);

	return a0;
}

uint32_t crc32_with_insn_b(const uint8_t *p, int len)
{
	uint32_t x = 0xffffffff;
	for (int i = 0; i < len; i++) {
		x = x ^ p[i];
		x = crc32_insn_b(x);
	}
	return ~x;
}

uint32_t crc32_with_insn_h(const uint16_t *p, int len)
{
	uint32_t x = 0xffffffff;
	for (int i = 0; i < len; i++) {
		x = x ^ p[i];
		x = crc32_insn_h(x);
	}
	return ~x;
}

uint32_t crc32_with_insn_w(const uint32_t *p, int len)
{
	uint32_t x = 0xffffffff;
	for (int i = 0; i < len; i++) {
		x = x ^ p[i];
		assert(crc32_insn_w_clmul(x) == crc32_insn_w(x));
		x = crc32_insn_w(x);
	}
	return ~x;
}

// -----------------------------------------------------------------------

int main()
{
	uint8_t s[] = "The quick brown fox jumps over the lazy dog ABCDEFGHIJKLMNOPQRST";
	int length = strlen((const char*)s);
	uint32_t good_crc32q = 0xA9DE0134;
	uint32_t good_crc32 = 0x1BE7DE66;
	uint32_t c;

	printf("length: %d\n", length);
	assert(length % 8 == 0);

	printf("----\n");

	printf("known_good_crc32q:   0x%08lX\n", (long)good_crc32q);

	c = crc32q_simple(s, length);
	printf("crc32q_simple_ref:   0x%08lX\n", (long)c);
	assert(c == good_crc32q);

	c = crc32q_clmul_simple((uint32_t*)s, length / 4);
	printf("crc32q_clmul_simple: 0x%08lX\n", (long)c);
	assert(c == good_crc32q);

	c = crc32q_clmul((uint64_t*)s, length / 8);
	printf("crc32q_with_clmul:   0x%08lX\n", (long)c);
	assert(c == good_crc32q);

	c = crc32q_clmul_u32((uint32_t*)s, length / 4);
	printf("crc32q_clmul_u32:    0x%08lX\n", (long)c);
	assert(c == good_crc32q);

	printf("----\n");

	printf("known_good_crc32:    0x%08lX\n", (long)good_crc32);

	c = crc32(0, s, length);
	printf("zlib_crc32_result:   0x%08lX\n", (long)c);
	assert(c == good_crc32);

	c = crc32_simple(s, length);
	printf("crc32_simple_ref:    0x%08lX\n", (long)c);
	assert(c == good_crc32);

	c = crc32_clmul((uint64_t*)s, length / 8);
	printf("crc32_with_clmul:    0x%08lX\n", (long)c);
	assert(c == good_crc32);

	c = crc32_with_insn_b((uint8_t*)s, length);
	printf("crc32_with_insn_b:   0x%08lX\n", (long)c);
	assert(c == good_crc32);

	c = crc32_with_insn_h((uint16_t*)s, length / 2);
	printf("crc32_with_insn_h:   0x%08lX\n", (long)c);
	assert(c == good_crc32);

	c = crc32_with_insn_w((uint32_t*)s, length / 4);
	printf("crc32_with_insn_w:   0x%08lX\n", (long)c);
	assert(c == good_crc32);

	return 0;
}
