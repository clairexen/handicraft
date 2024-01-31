#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>

// Marsaglia, George (July 2003). "Xorshift RNGs". Journal of Statistical Software. 8 (14).
// doi:10.18637/jss.v008.i14 https://www.jstatsoft.org/article/view/v008i14, page 4
static inline uint32_t xorshift32(uint32_t x)
{
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	return x;
}

int main()
{
	// brute-force demonstrator that xorshift32() has one non-trivial cycle of length 2**32-1
	uint8_t *maskdata = calloc(0x20000000, 1);
	for (uint32_t i = 1; i != 0; i++) {
		if (((maskdata[i/8] >> (i%8)) & 1) != 0)
			continue;
		printf("Cycle start at 0x%08x.\n", i);
		uint32_t cnt = 0, k = i;
		do {
			cnt++;
			k = xorshift32(k);
			// assert(((maskdata[k/8] >> (k%8)) & 1) == 0);
			maskdata[k/8] |= 1 << (k%8);
			if (__builtin_expect((cnt & ~0xfffffff) == cnt, 0))
				printf("  cycle length is 0x%08x and counting\n", cnt);
		} while (__builtin_expect(k != i, 1));
		printf("  cycle length is 0x%08x.\n", cnt);
	}
	free(maskdata);
	return 0;
}
