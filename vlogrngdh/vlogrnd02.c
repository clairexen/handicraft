#include "vlogrnd.h"
#include  <stdio.h>
#include  <stdint.h>

int main()
{
	long seed = 0;
	uint32_t buffer[1024*1024];
	int i;

	while (1) {
		for (i = 0; i < 1024*1024; i++) {
			uint32_t v1 = rtl_dist_uniform(&seed, UNIFORM_MIN, UNIFORM_MAX);
			uint32_t v2 = rtl_dist_uniform(&seed, UNIFORM_MIN, UNIFORM_MAX);
			v1 = (v1 % 371064217) & 0xffff;
			v2 = (v2 % 371064217) & 0xffff;
			buffer[i] = v1 | (v2 << 16);
		}
		fwrite(buffer, 1024*1024*sizeof(uint32_t), 1, stdout);
	}

	return 0;
}

