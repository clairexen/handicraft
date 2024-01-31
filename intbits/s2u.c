#include <stdint.h>
#include <assert.h>

void checker(uint64_t b)
{
	// Version that passes all cbmc checks
	uint64_t int64_max = ((uint64_t)1 << 63) - 1;
	int64_t bs = b > int64_max ? -1-(int64_t)(~b) : (int64_t)b;

	// trivial version
	int64_t bs2 = b;

	assert(bs == bs2);
}
