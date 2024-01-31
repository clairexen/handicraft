#include "Vtest017.h"

int main() {
	Vtest017 tb;
	tb.a = 31;
	tb.eval();
	printf("%d %s\n", tb.y, tb.y == 0 ? "OK" : "ERROR");
	tb.final();
	return 0;
}
