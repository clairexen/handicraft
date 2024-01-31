#include "Vtest015.h"

int main() {
	Vtest015 tb;
	tb.a = 0;
	tb.eval();
	for (int i = 3; i >= 0; i--)
		printf("%d", (tb.y >> i) & 1);
	printf("\n");
	tb.final();
	return 0;
}
