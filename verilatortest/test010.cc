#include "Vtest010.h"

int main() {
	Vtest010 tb;
	tb.eval();
	for (int i = 0; i < 5; i++)
		printf("%c", tb.y & (1 << (4-i)) ? '1' : '0');
	printf("\n");
	tb.final();
	return 0;
}
