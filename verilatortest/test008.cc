#include <string>
#include "Vtest008.h"

const char *good_results[] = {
	"0000000010000000",
	"1100000011000000",
	"1110000011100000",
	"1110000011100000",
	"1111000011110000",
	"1111000011110000",
	"1111000011110000",
	"1111000011110000",
	"1111100011111000",
	"1111100011111000",
	"1111100011111000",
	"1111100011111000",
	"1111100011111000",
	"1111100011111000",
	"1111100011111000",
	"1111100011111000"
};

int main() {
	Vtest008 tb;
	std::string buffer;
	for (int i = 0; i < 256; i++) {
		tb.a = i >> 3;
		tb.b = i;
		tb.eval();
		buffer += '0' + tb.y;
		if (i % 16 == 15) {
			printf("%s %s %s\n", buffer.c_str(), good_results[i/16],
					buffer == good_results[i/16] ? "ok" : "error");
			buffer.clear();
		}
	}
	tb.final();
	return 0;
}
