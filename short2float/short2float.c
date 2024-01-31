/*
 *  Copyright (C) 2019  Clifford Wolf <clifford@clifford.at>
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
 *
 *  gcc -Werror -Wall -Wextra -O2 -o short2float short2float.c && ./short2float
 *
 */

#define RVINTRIN_EMULATE
#include "rvintrin.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

typedef union {
	uint32_t i;
	float f;
} sf_t;

float short2float_f(int v)
{
	double ret = v;
	ret /= 32768;
	return ret;
}

uint32_t short2float(int16_t v)
{
	uint32_t a = (int32_t)v;
	uint32_t b = _rv32_max(a, -a);
	uint32_t c = _rv32_clz(b);
	uint32_t d = b << (c+1) >> 9;
	d |= ((143-c) & 255) << 23;
	d |= v >> 15 << 31;
	return _rv_cmov(v, d, 0);
}

uint32_t short2float_asm(int16_t v)
{
	uint32_t a0, a1, a2, a3;

	a0 = (int32_t)v;		// short2float:
	a1 = -a0;			//	neg a1, a0
	a3 = _rv32_max(a0, a1);		//	max a1, a1, a0
	a2 = _rv32_clz(a3);		//	clz a2, a1
	a0 = a0 >> 31;			//	srli a0, a0, 31
	a3 = a3 << a2;			//	sll a3, a1, a2
	a3 = a3 >> 15;			//	srli a3, a3, 15
	a2 = -a2;			//	neg a2, a2
	a2 += 143;			//	addi a2, a2, 143
	a0 = _rv32_packh(a2, a0);	//	packh a0, a2, a0
	a0 = _rv32_pack(a3, a0);	//	pack a0, a3, a0
	a0 <<= 7;			//	slli a4, a4, 7
	a1 = _rv32_orc(a1);		//	orc a1, a1
	a0 = a0 & a1;			//	and a0, a0, a1
	return a0;			//	ret
}

uint32_t short2float_asm2(int16_t v)
{
	uint32_t a0, a1, a2, a3;

	a0 = (int32_t)v;		// short2float:
	a1 = -a0;			//	neg a1, a0
	a1 = _rv32_max(a0, a1);		//	max a1, a1, a0
	a2 = _rv32_clz(a1);		//	clz a2, a1
	a3 = a1 << a2;			//	sll a3, a1, a2
	a3 = a3 << 1;			//	slli a3, a3, 1
	a2 = -a2;			//	neg a2, a2
	a2 += 143;			//	addi a2, a2, 143
	a3 = _rv32_fsr(a3, a2, 8);	//	fsri a3, a3, a2, 8
	a0 = a0 >> 31;			//	srli a0, a0, 31
	a0 = _rv32_fsr(a3, a0, 1);	//	fsri a0, a3, a0, 1
	a0 = _rv_cmov(a1, a0, 0);	//	cmov a0, a1, a0, zero
	return a0;			//	ret
}

int main()
{
	FILE *f = fopen("short2float.hex", "wt");
	for (int i = -32768; i < 32768; i++) {
		sf_t a;
		uint32_t b, c, d;
		a.f = short2float_f(i);
		b = short2float(i);
		c = short2float_asm(i);
		d = short2float_asm2(i);
		printf("checking %d (%f 0x%08x 0x%08x 0x%08x 0x%08x).\n", i, a.f, a.i, b, c, d);
		assert(a.i == b);
		assert(a.i == c);
		assert(a.i == d);
		fprintf(f, "%08x\n", a.i);
	}
	printf("OKAY\n");
	fclose(f);
	return 0;
}
