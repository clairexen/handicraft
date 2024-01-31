
#include <stdio.h>
#include <stdlib.h>
#include "libcurvetr.h"

int main(int argc, char **argv)
{
	libcurvetr_setup(argc < 2 ? "/dev/ttyUSB0" : argv[1]);

	int count = 0;
	double r = argc < 3 ? 33 : strtod(argv[2], NULL);
	double v[6], adc0, adc1, adc2, adc3, adc4;
	double u_be, u_ce, i;

	fprintf(stderr, "\n");
	fprintf(stderr, "Test Circuit:\n");
	fprintf(stderr, "=============\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "   POT1 a=VCC mid=ADC0 b=GND\n");
	fprintf(stderr, "   POT2 a=VCC mid=ADC1 b=GND\n");
	fprintf(stderr, "   NPN c=VCC b=ADC0 e=ADC2\n");
	fprintf(stderr, "   NPN c=VCC b=ADC1 e=ADC3\n");
	fprintf(stderr, "   NPN c=ADC3 b=ADC2 e=ADC4    <--- Unit under test\n");
	fprintf(stderr, "   R=%.0f a=ADC4 b=GND\n", r);
	fprintf(stderr, "\n");
	fprintf(stderr, "   (see npn.sch for qucs schematic drawing)\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "Use POT1 to adjust the base-emitter-voltage and\n");
	fprintf(stderr, "use POT2 to adjust the collector-emitter-voltage.\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "For currents up to 100 mA use something like R=33 Ohm.\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "Press ENTER to start recording.\n");

	while (!libcurvetr_gotkey())
	{
		libcurvetr_getvolt(v);
		adc0 = v[0], adc1 = v[1], adc2 = v[2], adc3=v[3], adc4=v[4];

		u_be = adc2 - adc4;
		u_ce = adc3 - adc4;
		i = adc4 / r;

		fprintf(stderr, "U_1=%5.3f/%5.3f U_2=%5.3f/%5.3f U_be=%5.3fV U_ce=%5.3fV I_c=%5.3fA   \r", adc0, adc2, adc1, adc3, u_be, u_ce, i);
	}

	fprintf(stderr, "Press ENTER to stop recording.\n");

	while (!libcurvetr_gotkey())
	{
		libcurvetr_getvolt(v);
		adc0 = v[0], adc1 = v[1], adc2 = v[2], adc3=v[3], adc4=v[4];

		u_be = adc2 - adc4;
		u_ce = adc3 - adc4;
		i = adc4 / r;

		fprintf(stderr, ".");
		printf("%f %f %f\n", u_be, u_ce, i);
		count++;
	}

	fprintf(stderr, "\nTotal nuber of u_be-u_ce-i-tuples: %d\n", count);
	libcurvetr_shutdown();
	return 0;
}

