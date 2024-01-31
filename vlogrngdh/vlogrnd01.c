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
			buffer[i] = rtl_dist_uniform(&seed, UNIFORM_MIN, UNIFORM_MAX);
		fwrite(buffer, 1024*1024*sizeof(uint32_t), 1, stdout);
	}

	return 0;
}

