#include "Vtest012.h"

int main() {
	Vtest012 tb;
	for (int i = 0; i < 8; i++) {
		tb.a = i;
		tb.eval();
		printf("%d %d\n", tb.a, tb.y);
	}
	tb.final();
	return 0;
}
