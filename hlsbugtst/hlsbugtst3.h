#ifndef TESTPRJ_H
#define TESTPRJ_H

#include <hls_stream.h>
#include <stdint.h>

void hlsbugtst3(hls::stream<uint32_t> &inA, hls::stream<uint32_t> &inB, hls::stream<uint32_t> &out);

#endif
