#include "Vtest007.h"

int main() {
	Vtest007 tb;
	for (int i = 0; i < 16; i++) {
		tb.a = i;
		tb.eval();
		printf("%x -> %02x  (%s)\n", tb.a, tb.y, (tb.y == ((i << 3) & 0x1f)) ? "ok" : "error");
	}
	tb.final();
	return 0;
}
