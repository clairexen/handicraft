
#include <stdio.h>
#include "libcurvetr.h"

int main(int argc, char **argv)
{
	libcurvetr_setup(argc < 2 ? "/dev/ttyUSB0" : argv[1]);
	while (!libcurvetr_gotkey()) {
		double v[6];
		libcurvetr_getvolt(v);
		printf("%10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\r", v[0], v[1], v[2], v[3], v[4], v[5]);
		fflush(stdout);
	}
	libcurvetr_shutdown();
	printf("\n");
	return 0;
}

