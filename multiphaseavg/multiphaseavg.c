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

#include "multiphaseavg.h"
#include <stdlib.h>
#include <complex.h>
#include <assert.h>
#include <math.h>

#define PI 3.14159265358979323846

static void radix2butterfly(int num_insamples, complex float *insamples_even, complex float *insamples_odd, complex float *outsamples)
{
	for (int i = 0; i < num_insamples; i++) {
		complex float omega = cexpf(-I * PI * i / num_insamples);
		outsamples[i] = insamples_even[i] + insamples_odd[i]*omega;
		outsamples[num_insamples + i] = insamples_even[i] - insamples_odd[i]*omega;
	}
}

static void radix2fft(int num_samples, complex float *samples)
{
	if (num_samples == 1)
		return;

	complex float *samples_even = malloc(sizeof(complex float) * num_samples / 2);
	complex float *samples_odd = malloc(sizeof(complex float) * num_samples / 2);

	for (int i = 0; i < num_samples/2; i++) {
		samples_even[i] = samples[2*i];
		samples_odd[i] = samples[2*i+1];
	}

	radix2fft(num_samples / 2, samples_even);
	radix2fft(num_samples / 2, samples_odd);
	radix2butterfly(num_samples / 2, samples_even, samples_odd, samples);

	free(samples_even);
	free(samples_odd);
}

void multiphaseavg(int num_samples, int num_waves, int oversampling,
		float **insamples, float *inphases, float *outsamples)
{
	// num_samples and oversampling must be powers of two
	assert((num_samples & (num_samples-1)) == 0);
	assert((oversampling & (oversampling-1)) == 0);

	// many forward DFTs
	complex float **samples_dft = malloc(sizeof(complex float*) * num_waves);
	for (int i = 0; i < num_waves; i++) {
		samples_dft[i] = malloc(sizeof(complex float) * num_samples);
		for (int k = 0; k < num_samples; k++)
			samples_dft[i][k] = insamples[i][k];
		radix2fft(num_samples, samples_dft[i]);
	}

	// correct phases and average in buckets
	complex float **bucket_dft = malloc(sizeof(complex float*) * oversampling);
	int *bucket_cnt = malloc(sizeof(int) * oversampling);

	for (int i = 0; i < oversampling; i++) {
		bucket_dft[i] = malloc(sizeof(complex float) * num_samples);
		for (int k = 0; k < num_samples; k++)
			bucket_dft[i][k] = 0;
		bucket_cnt[i] = 0;
	}

	for (int i = 0; i < num_waves; i++) {
		int b = (int)(inphases[i]*oversampling + oversampling + 0.5) % oversampling;
		float p = inphases[i] - (float)b / oversampling;
		for (int k = 0; k < num_samples; k++) {
			float freq = (float)(k < num_samples/2 ? k : -num_samples + k) / num_samples;
			complex float omega = cexpf(-2 * I * PI * p * freq);
			bucket_dft[b][k] += samples_dft[i][k] * omega;
		}
		bucket_cnt[b]++;
	}

	for (int i = 0; i < oversampling; i++) {
		if (bucket_cnt[i] > 0)
			for (int k = 0; k < num_samples; k++)
				bucket_dft[i][k] /= bucket_cnt[i];
	}

	// combine individual FFTs using butterflies
	int new_num_samples = num_samples;
	for (int s = oversampling/2; s > 0; s /= 2) {
		new_num_samples = 2 * new_num_samples;
		for (int i = 0; i < s; i++) {
			complex float *new_samples = malloc(sizeof(complex float) * new_num_samples);
			radix2butterfly(new_num_samples / 2, bucket_dft[i], bucket_dft[i+s], new_samples);
			free(bucket_dft[i]);
			bucket_dft[i] = new_samples;
		}
	}
	assert(new_num_samples == num_samples * oversampling);

	// obtain oversampled time-domain representation using an inverse FFT
	for (int i = 0; i < new_num_samples; i++)
		bucket_dft[0][i] = conjf(bucket_dft[0][i]);
	radix2fft(new_num_samples, bucket_dft[0]);
	for (int i = 0; i < new_num_samples; i++)
		outsamples[i] = crealf(bucket_dft[0][i]) / new_num_samples;

	// free heap memory
	for (int i = 0; i < num_waves; i++)
		free(samples_dft[i]);
	for (int i = 0; i < oversampling; i++)
		free(bucket_dft[i]);
	free(samples_dft);
	free(bucket_dft);
	free(bucket_cnt);
}
