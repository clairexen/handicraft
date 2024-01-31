#include "vivadobug04.h"

uint32_t vivadobug04(uint32_t x32)
{
	x32 ^= x32 << 13;
	x32 ^= x32 >> 17;
	x32 ^= x32 << 5;
	return x32;
}

