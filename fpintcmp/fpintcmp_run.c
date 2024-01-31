#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

uint32_t x = 123456789;
uint32_t y = 362436069;
uint32_t z = 521288629;
uint32_t w = 88675123;

uint32_t xorshift128()
{
	uint32_t t = x ^ (x << 11);
	x = y; y = z; z = w;
	w ^= (w >> 19) ^ t ^ (t >> 8);
	return w;
}

int main()
{
	for (int i = 0; i < 100000000; i++)
	{
		uint32_t ai = xorshift128();
		uint32_t bi = xorshift128();

		// clear sign bit
		ai &= 0x7fffffff;
		bi &= 0x7fffffff;

		// no nan/inf values
		if ((ai & 0x7f800000) == 0x7f800000) continue;
		if ((bi & 0x7f800000) == 0x7f800000) continue;

		float af = *(float*)&ai;
		float bf = *(float*)&bi;

		bool x = ai < bi;
		bool y = af < bf;

		if (x != y)
			printf("%d %d - %08x %08x - %+e %+e\n", x, y, ai, bi, af, bf);
	}
	return 0;
}
