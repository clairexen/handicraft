#include "libvc830.h"
#include <stdio.h>

int main(int argc, char **argv)
{
	VC830 *dmm = VC830::openDev(argc >= 2 ? argv[1] : NULL);
	if (dmm == NULL)
		return 1;
	while (1) {
		if (dmm->update() < 0)
			break;
		dmm->print(stdout);
		fflush(stdout);
	}
	delete dmm;
	return 0;
}

