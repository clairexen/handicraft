#include "Vtest004.h"

int main() {
	Vtest004 tb;
	tb.a = 0;
	tb.eval();
	printf("%08x%s\n", tb.y, tb.y == 0x00000000 ? "" : " <- error");
	tb.a = 1;
	tb.eval();
	printf("%08x%s\n", tb.y, tb.y == 0x010000ff ? "" : " <- error");
	tb.final();
	return 0;
}
