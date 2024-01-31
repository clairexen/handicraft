
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
	int w, h, i, j;

	for (w=metaleds_width, h=metaleds_height; w>40 && h>10; w-=4, h-=4)
	{
		for (i=0; i<w; i++) {
			metaleds_setpixel(frame, metaleds_width/2-w/2+i, metaleds_height/2-h/2, 255);
			metaleds_setpixel(frame, metaleds_width/2-w/2+i, metaleds_height/2+h/2-1, 255);
		}
		for (i=0; i<h; i++) {
			metaleds_setpixel(frame, metaleds_width/2-w/2, metaleds_height/2-h/2+i, 255);
			metaleds_setpixel(frame, metaleds_width/2+w/2-1, metaleds_height/2-h/2+i, 255);
		}
			
	}

	for (i=0; i<16; i++)
	{
		for (j=0; j<h-4; j++)
			metaleds_setpixel(frame, metaleds_width/2-16+i*2, metaleds_height/2-h/2+j+2, i << 4);
		metaleds_setpixel(frame, metaleds_width/2-16+i*2, metaleds_height/2-h/2, 255);
		metaleds_setpixel(frame, metaleds_width/2-16+i*2, metaleds_height/2+h/2-1, 255);
	}

	while (1)
	{
		if (metaleds_frame(frame) < 0) {
			fprintf(stderr, "MetaLEDs frame update failed!\n");
			return 1;
		}
		sleep(1);
	}

	metaleds_free(frame);
	return 0;
}

