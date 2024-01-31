// This is free and unencumbered software released into the public domain.
// 
// Anyone is free to copy, modify, publish, use, compile, sell, or
// distribute this software, either in source code form or as a compiled
// binary, for any purpose, commercial or non-commercial, and by any
// means.
// 
// -------------------------------------------------------
// Written by Clifford Wolf <clifford@clifford.at> in 2016
// -------------------------------------------------------

#ifndef MULTIPHASEAVG_H
#define MULTIPHASEAVG_H

#ifdef __cplusplus
extern "C" {
#endif

void multiphaseavg(int num_samples, int num_waves, int oversampling,
		float **insamples, float *inphases, float *outsamples);

#ifdef __cplusplus
}
#endif

#endif
