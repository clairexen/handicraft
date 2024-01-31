#include "Vtest006.h"

int main() {
	Vtest006 tb;
	for (int i = 0; i < 16; i++) {
		tb.a = i;
		tb.eval();
		printf("%x -> %02x  (%s)\n", tb.a, tb.y, (tb.y == i+8) ? "ok" : "error");
	}
	tb.final();
	return 0;
}
