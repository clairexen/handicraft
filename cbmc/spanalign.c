#include <stdint.h>
#include <stdbool.h>

// check if a span of length 1, 2, 4, or 8, given by first and last element, is
// naturally aligned to addresses divisible by the span length.
bool refimpl(uint32_t first, uint32_t last)
{
	switch (last-first+1)
	{
	case 2:
		return (first & 1) == 0;
	case 4:
		return (first & 3) == 0;
	case 8:
		return (first & 7) == 0;
	default:
		return true;
	}
}

// same
bool myimpl_1(uint32_t first, uint32_t last)
{
	return (last ^ first) == (last - first);
}

void test(uint32_t first, uint32_t last)
{
	if (last >= first && (last == first || last == first+1 || last == first+3 || last == first+7)) {
		assert(refimpl(first, last) == myimpl_1(first, last));
	}
}
