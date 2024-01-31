// See https://twitter.com/pasiphae_goals/status/1053512914466934784
// cbmc --trace --function check iseven.c

#include <stdbool.h>
#include <assert.h>

bool isEven(unsigned char x)
{
	for (int i = 0; i < 7; i++) {
		if (x == 0)
			return true;
		if (x == 1)
			return false;
		x *= x;
	}
}

void check(unsigned char x)
{
	bool is_even_1 = isEven(x);
	bool is_even_2 = !(x & 1);
	assert(is_even_1 == is_even_2);
}
