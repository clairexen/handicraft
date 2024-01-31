#include "Vtest019.h"

int main() {
	Vtest019 tb;
	for (int a = 0; a < 16; a++) {
		tb.a = a;
		tb.eval();
		printf("a = %2d -> y = %2d  %s\n", tb.a, tb.y, tb.y == 15 ? "OK" : "ERROR");
	}
	tb.final();
	return 0;
}
