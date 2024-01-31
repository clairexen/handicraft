#include "hlsbugtst3.h"
#include <stdio.h>

int main()
{
	hls::stream<uint32_t> inA, inB, out;
	int nok_counter = 0;
	int counter = 0;

	for (int i = 0; i < 20; i++) {
		inA.write(i);
		inB.write(12344 * i);
	}

	while (counter < 20)
	{
		hlsbugtst3(inA, inB, out);

		while (!out.empty()) {
			uint32_t val = out.read();
			uint32_t ref = 12345 * counter;
			printf("%2d: 0x%08x 0x%08x %s\n", counter, val, ref, val == ref ? "OK" : "NOK");
			nok_counter += val != ref;
			counter++;
		}
	}

	if (nok_counter) {
		printf("Error: got %d NOKs!\n");
		return 1;
	}

	printf("PASSED.\n");
	return 0;
}
