// cbmc --trace --function lastblock_check lastblock.c
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#define ctz_zero_is_undef

uint32_t ctz(uint32_t x)
{
	for (int count = 0; count < 32; count++)
		if ((x >> count) & 1)
			return count;
#ifdef ctz_zero_is_undef
	assert(0);
#endif
	return 32;
}

uint32_t lastblock_ref(uint32_t x)
{
	return (x & ~(x + (x & -x)));
}

// https://twitter.com/babbageboole/status/990028009154400256
uint32_t lastblock_knuth(uint32_t x)
{
	return (x & ~((x | (x - 1)) + 1));
}

// https://twitter.com/oe1cxw/status/989918504991252487
uint32_t lastblock_ctz(uint32_t x)
{
	int a = ctz(x);
	return ((1 << ctz(~(x >> a))) - 1) << a;
}

void lastblock_check(uint32_t x)
{
	uint32_t p = lastblock_ref(x);
	uint32_t q = lastblock_knuth(x);
	assert(p == q);

#ifdef ctz_zero_is_undef
	if (x == 0xffffffff) return;
#endif
	if (x) {
		q = lastblock_ctz(x);
		assert(p == q);
	}
}

int main()
{
	uint32_t x = 0xffffffff, y = lastblock_ref(x);
	printf("0x%08x 0x%08x\n", x, y);
	return 0;
}
