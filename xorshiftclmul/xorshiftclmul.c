// cbmc --trace --function test xorshiftclmul.c
#include <stdint.h>

uint32_t clmul(uint32_t rs1, uint32_t rs2)
{
	uint32_t x = 0;
	for (int i = 0; i < 32; i++)
		if ((rs2 >> i) & 1)
			x ^= rs1 << i;
	return x;
}

// Marsaglia, George (July 2003). "Xorshift RNGs". Journal of Statistical Software. 8 (14).
// doi:10.18637/jss.v008.i14 https://www.jstatsoft.org/article/view/v008i14, page 4
uint32_t xorshift32(uint32_t x)
{
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	return x;
}

// naive inverse
uint32_t xorshift32_inv(uint32_t x)
{
	uint32_t t;
	t = x ^ (x << 5);
	t = x ^ (t << 5);
	t = x ^ (t << 5);
	t = x ^ (t << 5);
	t = x ^ (t << 5);
	x = x ^ (t << 5);
	x = x ^ (x >> 17);
	t = x ^ (x << 13);
	x = x ^ (t << 13);
	return x;
}

// clmul inverse
uint32_t xorshift32_inv_clmul(uint32_t x)
{
	x = clmul(x, 0x42108421);
	x = x ^ (x >> 17);
	x = clmul(x, 0x04002001);
	return x;
}

void test(uint32_t x)
{
	uint32_t y = xorshift32(x);
	assert(x == xorshift32_inv(y));
	assert(x == xorshift32_inv_clmul(y));
}
