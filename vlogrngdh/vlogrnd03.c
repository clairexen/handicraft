#include "vlogrnd.h"
#include  <stdio.h>
#include  <stdint.h>

int main()
{
	long seed = 0;
	uint32_t buffer[1024*1024];
	int i;

	while (1) {
		for (i = 0; i < 1024*1024; i++)
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

			buffer[i] = x;
		}
		fwrite(buffer, 1024*1024*sizeof(uint32_t), 1, stdout);
	}

	return 0;
}

