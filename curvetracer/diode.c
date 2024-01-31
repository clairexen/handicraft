
#include <stdio.h>
#include <stdlib.h>
#include "libcurvetr.h"

int main(int argc, char **argv)
{
	libcurvetr_setup(argc < 2 ? "/dev/ttyUSB0" : argv[1]);

	int count = 0;
	double r = argc < 3 ? 33 : strtod(argv[2], NULL);
	double v[6], adc0, adc2, adc4;
	double u, i;

	fprintf(stderr, "\n");
	fprintf(stderr, "Test Circuit:\n");
	fprintf(stderr, "=============\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "   POT a=VCC mid=ADC0 b=GND\n");
	fprintf(stderr, "   NPN c=VCC b=ADC0 e=ADC2\n");
	fprintf(stderr, "   DIODE a=ADC2 k=ADC4          <--- Unit under test\n");
	fprintf(stderr, "   R=%.0f a=ADC4 b=GND\n", r);
	fprintf(stderr, "\n");
	fprintf(stderr, "   (see diode.sch for qucs schematic drawing)\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "For currents up to 100 mA use something like R=33 Ohm.\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "Press ENTER to stop recording.\n");

	while (!libcurvetr_gotkey())
	{
		libcurvetr_getvolt(v);
		adc0 = v[0], adc2 = v[2], adc4 = v[4];

		u = adc2 - adc4;
		i = adc4 / r;

		fprintf(stderr, ".");
		printf("%f %f\n", u, i);
		count++;
	}

	fprintf(stderr, "\nTotal nuber of u-i-tuples: %d\n", count);
	libcurvetr_shutdown();
	return 0;
}

