
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "libmetaleds.h"

int main(int argc, char **argv)
{
	if (metaleds_init(argc == 2 ? argv[1] : "sim:72:48") < 0) {
		fprintf(stderr, "MetaLEDs initialization failed!\n");
		return 1;
	}

	metaleds_frame_p frame = metaleds_malloc();
	int x, y;

	while (1)
	{
		for (y=0; y<metaleds_height; y++)
		{
			metaleds_clrsrc(frame);
			for (x=0; x<metaleds_width; x++)
				metaleds_setpixel(frame, x, y, 255);
			if (metaleds_frame(frame) < 0) {
				fprintf(stderr, "MetaLEDs frame update failed!\n");
				goto exit;
			}
			sleep(1);
		}
		for (x=0; x<metaleds_width; x++)
		{
			metaleds_clrsrc(frame);
			for (y=0; y<metaleds_height; y++)
				metaleds_setpixel(frame, x, y, 255);
			if (metaleds_frame(frame) < 0) {
				fprintf(stderr, "MetaLEDs frame update failed!\n");
				goto exit;
			}
			sleep(1);
		}
	}

exit:
	metaleds_free(frame);
	return 0;
}

