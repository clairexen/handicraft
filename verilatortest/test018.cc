#include "Vtest018.h"

int main() {
	Vtest018 tb;
	for (int a = 0; a < 16; a++)
	for (int b = 0; b < 64; b++) {
		int y = b < 32 ? ((a << b) & 15) : 0;
		tb.a = a;
		tb.b = b;
		tb.eval();
		if (tb.y != y)
			printf("ERROR: a = %2d, b = %2d -> y = %2d instead of y = %2d\n", tb.a, tb.b, tb.y, y);
	}
	tb.final();
	return 0;
}
