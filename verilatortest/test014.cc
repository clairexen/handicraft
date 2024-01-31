#include "Vtest014.h"

int main() {
	Vtest014 tb;
	tb.a = 128;
	tb.eval();
	printf("%d\n", tb.y);
	tb.final();
	return 0;
}
