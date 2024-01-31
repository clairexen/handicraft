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
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int main()
{
	int num_waves = 1024;
	int num_samples = 16;
	int oversampling = 32;

	float **insamples = malloc(sizeof(float*) * num_waves);
	float *inphases = malloc(sizeof(float) * num_waves);

	int ch, rc;

	for (int i = 0; i < num_waves; i++)
	{
		do ch = getchar(); while (ch == '\n');
		assert(ch == 'I');

		rc = scanf("%f", &inphases[i]);
		assert(rc == 1);

		insamples[i] = malloc(sizeof(float) * num_samples);
		for (int k = 0; k < num_samples; k++) {
			rc = scanf("%f", &insamples[i][k]);
			assert(rc == 1);
		}
	}

	float *expected_outsamples = malloc(sizeof(float) * num_samples * oversampling);
	float *outsamples = malloc(sizeof(float) * num_samples * oversampling);

	do ch = getchar(); while (ch == '\n');
	assert(ch == 'O');

	for (int i = 0; i < num_samples * oversampling; i++) {
		rc = scanf("%f", &expected_outsamples[i]);
		assert(rc == 1);
	}

	multiphaseavg(num_samples, num_waves, oversampling, insamples, inphases, outsamples);

	float rms_error = 0;
	for (int i = 0; i < num_samples * oversampling; i++) {
		printf("%+.6f %+.6f %.6f\n", expected_outsamples[i], outsamples[i], fabsf(expected_outsamples[i] - outsamples[i]));
		rms_error += (expected_outsamples[i] - outsamples[i]) * (expected_outsamples[i] - outsamples[i]);
	}

	printf("rms_error=%e\n", sqrtf(rms_error));

	for (int i = 0; i < num_waves; i++)
		free(insamples[i]);
	free(insamples);
	free(inphases);
	free(expected_outsamples);
	free(outsamples);
	return 0;
}

