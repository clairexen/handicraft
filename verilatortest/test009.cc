#include "Vtest009.h"

int main() {
	Vtest009 tb;
	tb.eval();
	printf("%d%d%d%d%d%d%d\n", tb.a, tb.b, tb.c, tb.d, tb.e, tb.f, tb.g);
	tb.final();
	return 0;
}
