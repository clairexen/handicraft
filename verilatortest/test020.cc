#include "Vtest020.h"

int main() {
	Vtest020 tb;
	for (int i = 0; i < 8; i++) {
		tb.a = i;
		tb.eval();
		printf("%d %c%c%c%c\n", i,
				tb.y & 8 ? '1' : '0', tb.y & 4 ? '1' : '0',
				tb.y & 2 ? '1' : '0', tb.y & 1 ? '1' : '0');
	}
	tb.final();
	return 0;
}
