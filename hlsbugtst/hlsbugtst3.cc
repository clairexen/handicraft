#include "hlsbugtst3.h"

void stream_buffer(hls::stream<uint32_t> &in, hls::stream<uint32_t> &out)
{
	if (!in.empty() && !out.full())
		out.write(in.read());
}

void stream_adder(hls::stream<uint32_t> &inA, hls::stream<uint32_t> &inB, hls::stream<uint32_t> &out)
{
	static int state = 0;
	static uint32_t acc;

	switch (state)
	{
	case 0:
		if (!inA.empty()) {
			acc = inA.read();
			state = 1;
		}
		break;
	case 1:
		if (!inB.empty()) {
			acc += inB.read();
			state = 2;
		}
		break;
	case 2:
		if (!out.full()) {
			out.write(acc);
			state = 0;
		}
		break;
	}
}

void hlsbugtst3(hls::stream<uint32_t> &inA, hls::stream<uint32_t> &inB, hls::stream<uint32_t> &out)
{
#pragma HLS dataflow

#pragma HLS INTERFACE port=inA axis
#pragma HLS INTERFACE port=inB axis
#pragma HLS INTERFACE port=out axis

	static hls::stream<uint32_t> bufA;
#pragma HLS stream variable=bufA depth=2

	static hls::stream<uint32_t> bufB;
#pragma HLS stream variable=bufB depth=2

	stream_buffer(inA, bufA);
	stream_buffer(inB, bufB);
	stream_adder(bufA, bufB, out);
}
