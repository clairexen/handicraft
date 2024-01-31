
#include "libcurvetr.h"

int main(int argc, char **argv)
{
	libcurvetr_setup(argc < 2 ? "/dev/ttyUSB0" : argv[1]);
	libcurvetr_calib();
	libcurvetr_shutdown();
	return 0;
}

