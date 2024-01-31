#include "vlogrnd.h"
#include  <stdio.h>
#include  <stdint.h>

int main()
{
	long seed = 0;
	int i;

	for (i = 0; i < 3; i++)
	{
		uint32_t v1 = rtl_dist_uniform(&seed, UNIFORM_MIN, UNIFORM_MAX);
		uint32_t v2 = rtl_dist_uniform(&seed, UNIFORM_MIN, UNIFORM_MAX);
		uint32_t x = 0;
		
		x ^= v1;
		x ^= x << 13;
		x ^= x >> 17;
		x ^= x << 5;

		x ^= v2;
		x ^= x << 13;
		x ^= x >> 17;
		x ^= x << 5;

		printf("%08x\n", (int)x);
	}

	return 0;
}

