#include <stdio.h>
#include <stdint.h>

uint32_t xorshift32()
{
	static uint32_t x32 = 314159265;
	x32 ^= x32 << 13;
	x32 ^= x32 >> 17;
	x32 ^= x32 << 5;
	return x32;
}

#if 0
uint32_t bext(uint32_t v, uint32_t mask)
{
	uint32_t c = 0, m = 1;
	while (mask) {
		uint32_t b = mask & -mask;
		if (v & b) c |= m;
		mask -= b;
		m <<= 1;
	}
	return c;
}

uint32_t bdep(uint32_t v, uint32_t mask)
{
	uint32_t c = 0, m = 1;
	while (mask) {
		uint32_t b = mask & -mask;
		if (v & m) c |= b;
		mask -= b;
		m <<= 1;
	}
	return c;
}

uint32_t bextdep(uint32_t v, uint32_t emask, uint32_t dmask)
{
	uint32_t c = 0;
	while (emask && dmask) {
		uint32_t eb = emask & -emask;
		uint32_t db = dmask & -dmask;
		if (v & eb) c |= db;
		emask -= eb;
		dmask -= db;
	}
	return c;
}

uint32_t perm_lut_bextdep(uint32_t lut, int a, int b)
{
	static const uint32_t masks[5][2] = {
		{0x55555555, 0xAAAAAAAA},
		{0x33333333, 0xCCCCCCCC},
		{0x0F0F0F0F, 0xF0F0F0F0},
		{0x00FF00FF, 0xFF00FF00},
		{0x0000FFFF, 0xFFFF0000}
	};

	uint32_t mask_a_notb = masks[a][1] & masks[b][0];
	uint32_t mask_nota_b = masks[a][0] & masks[b][1];

	uint32_t perm = lut & ~(mask_a_notb | mask_nota_b);

#if 0
	perm |= bdep(bext(lut, mask_a_notb), mask_nota_b);
	perm |= bdep(bext(lut, mask_nota_b), mask_a_notb);
#else
	perm |= bextdep(lut, mask_a_notb, mask_nota_b);
	perm |= bextdep(lut, mask_nota_b, mask_a_notb);
#endif

	return perm;
}
#endif

uint32_t perm_lut(uint32_t lut, int a, int b)
{
	static const uint32_t masks[5] = {
		0xAAAAAAAA, 0xCCCCCCCC, 0xF0F0F0F0, 0xFF00FF00, 0xFFFF0000
	};

	uint32_t mask_a_notb = masks[a] & ~masks[b];
	uint32_t mask_b_nota = masks[b] & ~masks[a];

	uint32_t perm = lut & ~(mask_a_notb | mask_b_nota);

	uint32_t lut_a_notb = lut & mask_a_notb;
	uint32_t lut_b_nota = lut & mask_b_nota;

	int shamt = ((1 << a) - (1 << b)) & 31;

	perm |= (lut_a_notb >> shamt) | (lut_a_notb << (-shamt & 31));
	perm |= (lut_b_nota << shamt) | (lut_b_nota >> (-shamt & 31));

	return perm;
}

uint32_t perm_lut_naive(uint32_t lut, int a, int b)
{
	uint32_t perm = 0;

	for (int p = 0; p < 32; p++)
	{
		int q = p & ~((1 << a) | (1 << b));
		q |= ((p >> a) & 1) << b;
		q |= ((p >> b) & 1) << a;

		if ((lut >> p) & 1)
			perm |= 1 << q;
	}

	return perm;
}

int main()
{
	int errcount = 0;

	for (int i = 0; i < 100; i++)
	{
		int a = xorshift32() % 5;
		int b = xorshift32() % 5;

		printf("--\n");
		printf("Swapped inputs: %d %d\n", a, b);

		uint32_t lut = xorshift32();
		uint32_t perm = perm_lut(lut, a, b);
		uint32_t perm_naive = perm_lut_naive(lut, a, b);

		printf("LUT: %08x %08x %s\n", lut, perm_lut(perm, a, b), lut == perm_lut(perm, a, b) ? "ok" : "ERROR");

		if (lut != perm_lut(perm, a, b))
			errcount++;

		printf("PERM: %08x %08x %s\n", perm, perm_naive, perm == perm_naive ? "ok" : "ERROR");

		if (perm != perm_naive)
			errcount++;

		for (int p = 0; p < 32; p++)
		{
			int q = p & ~((1 << a) | (1 << b));
			q |= ((p >> a) & 1) << b;
			q |= ((p >> b) & 1) << a;
			
			int lut_result = (lut >> p) & 1;
			int perm_result = (perm >> q) & 1;

			printf("%3d %2d -> %d %d %s\n", p, q, lut_result, perm_result,
					lut_result == perm_result ? "ok" : "ERROR");

			if (lut_result != perm_result)
				errcount++;
		}

		if (errcount != 0) {
			printf("%d Errors!\n", errcount);
			return 1;
		}
	}

	printf("--\n");
	printf("OK\n");
	return 0;
}
