#include "Vtest011.h"

int main() {
	Vtest011 tb;

#if 1
	tb.a0 = 0;
	tb.a1 = 0;
	tb.a2 = 0;
	tb.a3 = 0;
	tb.eval();

	for (int i = 37; i >= 34; i--)
		printf("%c", ((tb.y >> i) & 1) ? '1' : '0');
	printf("\n");
#endif

	tb.a0 = 15;
	tb.a1 = 15;
	tb.a2 = 15;
	tb.a3 = 15;
	tb.eval();

	for (int i = 37; i >= 34; i--)
		printf("%c", ((tb.y >> i) & 1) ? '1' : '0');
	printf("\n");

	tb.final();
	return 0;
}
