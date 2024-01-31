#ifndef DEMO_H
#define DEMO_H

#include <stdint.h>
#include <hls_stream.h>

void demo(
		hls::stream<uint32_t> &in_stream,
		hls::stream<uint32_t> &out_stream);

#endif
