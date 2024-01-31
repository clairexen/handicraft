#include "Vtest005.h"

int main() {
	Vtest005 tb;
	for (int i = 0; i < 16; i++) {
		tb.a = i;
		tb.eval();
		printf("%x -> %x  (%s)\n", tb.a, tb.y, (tb.y == 0 || tb.y == 1) ? "ok" : "error");
	}
	tb.final();
	return 0;
}
