#include "vivadobug04.h"

int main()
{
	uint32_t x32 = 123456789;
	for (int i = 0; i < 16; i++) {
		x32 = vivadobug04(x32);
		printf("%08x\n", x32);
	}
	return 0;
}
