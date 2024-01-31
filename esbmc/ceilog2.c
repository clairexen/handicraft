// Verify with: esbmc ceilog2.c

#include <stdio.h>
#include <math.h>

int ceil_log2(int x)
{
	if (x <= 0)
		return 0;

	int y = (x & (x - 1));
	y = (y | -y) >> 31;

	x |= (x >> 1);
	x |= (x >> 2);
	x |= (x >> 4);
	x |= (x >> 8);
	x |= (x >> 16);

	x >>= 1;
	x -= ((x >> 1) & 0x55555555);
	x = (((x >> 2) & 0x33333333) + (x & 0x33333333));
	x = (((x >> 4) + x) & 0x0f0f0f0f);
	x += (x >> 8);
	x += (x >> 16);
	x = x & 0x0000003f;

	return x - y;
}

int ceil_log2_ref(int x)
{
	if (x <= 0)
		return 0;

	for (int i = 0; i < 32; i++)
		if (((x-1) >> i) == 0)
			return i;

	__ESBMC_assert(0, "never reached");
}

int main()
{
#if 1
	int x = nondet_int();
	int x1 = ceil_log2(x);
	int x2 = ceil_log2_ref(x);
	__ESBMC_assert(x1 == x2, "x1 == x2");
#else
	int x = 1234;
	int x1 = ceil_log2(x);
	int x2 = ceil_log2_ref(x);
	int x3 = ceil(log2(x));
	printf("%d %d %d %d\n", x, x1, x2, x3);
#endif
	return 0;
}
