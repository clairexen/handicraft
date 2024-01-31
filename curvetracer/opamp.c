
#include <stdio.h>
#include <stdlib.h>
#include "libcurvetr.h"

int main(int argc, char **argv)
{
	libcurvetr_setup(argc < 2 ? "/dev/ttyUSB0" : argv[1]);

	int count = 0;
	double f = argc < 3 ? 1.0 : strtod(argv[2], NULL);
	double v[6], adc0, adc1, adc2;

	fprintf(stderr, "\n");
	fprintf(stderr, "Test Circuit:\n");
	fprintf(stderr, "=============\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "   ADC0: inverted input\n");
	fprintf(stderr, "   ADC1: non-inverted input\n");
	fprintf(stderr, "   ADC2: output\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "Press ENTER to start recording.\n");

	while (!libcurvetr_gotkey())
	{
		libcurvetr_getvolt(v);
		adc0 = v[0]*f, adc1 = v[1]*f, adc2 = v[2]*f;
		fprintf(stderr, "In-=%+6.3fV In+=%+6.3fV In_Diff=%+6.3fV Out=%+6.3fV    \r", adc0, adc1, adc1-adc0, adc2);
	}

	fprintf(stderr, "Press ENTER to stop recording.\n");

	while (!libcurvetr_gotkey())
	{
		libcurvetr_getvolt(v);
		adc0 = v[0]*f, adc1 = v[1]*f, adc2 = v[2]*f;

		fprintf(stderr, "<%d> In-=%+6.3fV In+=%+6.3fV In_Diff=%+6.3fV Out=%+6.3fV    \r", count, adc0, adc1, adc1-adc0, adc2);
		printf("%f %f %f %f\n", adc0, adc1, adc1-adc0, adc2);
		count++;
	}

	fprintf(stderr, "\nTotal nuber of tuples: %d\n", count);
	libcurvetr_shutdown();
	return 0;
}

