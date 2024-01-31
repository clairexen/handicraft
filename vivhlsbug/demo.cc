#include "demo.h"

void demo_func1(
		hls::stream<uint32_t> &in_stream,
		hls::stream<uint32_t> &out_stream)
{
#pragma HLS pipeline ii=1

	if (in_stream.empty())
		return;

	uint32_t value = in_stream.read();

	if ((value & 3) == 0)
		out_stream.write(value);
}

void demo_func2(
		hls::stream<uint32_t> &in_stream,
		hls::stream<uint32_t> &out_stream)
{
	static int state = 0;
	static uint32_t value;

#pragma HLS pipeline ii=1
#pragma HLS reset variable=state

	if (state == 0)
	{
		if (in_stream.empty())
			return;

		value = in_stream.read();
	}

	value ^= value << 13;
	value ^= value >> 17;
	value ^= value << 5;

	if (state == 3) {
		out_stream.write(value);
		state = 0;
	} else {
		state++;
	}
}

void demo(
		hls::stream<uint32_t> &in_stream,
		hls::stream<uint32_t> &out_stream)
{
	static hls::stream<uint32_t> internal_stream;

#pragma HLS dataflow
#pragma HLS interface ap_ctrl_none port=return
#pragma HLS interface port=in_stream axis
#pragma HLS interface port=out_stream axis
#pragma HLS stream variable=internal_stream depth=2

	demo_func1(in_stream, internal_stream);
	demo_func2(internal_stream, out_stream);
}
