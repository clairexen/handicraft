#include <stdio.h>
#include "demo.h"

uint32_t xorshifter(uint32_t value)
{
	for (int i = 0; i < 4; i++)	{
	  value ^= value << 13;
	  value ^= value >> 17;
	  value ^= value << 5;
	}
	return value;
}

void reference_impl(
		uint32_t value,
		hls::stream<uint32_t> &out_stream)
{
	if ((value & 3) == 0)
		out_stream.write(xorshifter(value));
}

int main()
{
	hls::stream<uint32_t> in_stream;
	hls::stream<uint32_t> out_stream;
	hls::stream<uint32_t> ref_out_stream;

	for (int i = 0; i < 100; i++) {
		printf("0%08x\n", i);
		in_stream.write(i);
		reference_impl(i, ref_out_stream);
	}

	for (int i = 0; i < 100; i += 4) {
		printf("0%08x\n", i);
		in_stream.write(i);
		reference_impl(i, ref_out_stream);
	}

	for (int i = 0; i < 100; i++) {
		printf("0%08x\n", i);
		in_stream.write(i);
		reference_impl(i, ref_out_stream);
	}

	int timeout = 100;

	while (!in_stream.empty() || timeout--)
	{
		demo(in_stream, out_stream);

		while (!out_stream.empty())
		{
			if (ref_out_stream.empty()) {
				printf("Error: Unexpected EOF in ref_out_stream!\n");
				return 1;
			}

			uint32_t value = out_stream.read();
			uint32_t ref_value = ref_out_stream.read();

			printf("1%08x\n", value);

			if (value != ref_value) {
				printf("Error: Mismatch!\n");
				return 1;
			}
		}
	}

	if (!ref_out_stream.empty()) {
		printf("Error: Unexpected EOF in out_stream!\n");
		return 1;
	}

	printf("OK\n");
	return 0;
}
