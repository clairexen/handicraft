
#ifndef LIBMETALEDS_H
#define LIBMETALEDS_H

extern int metaleds_width;
extern int metaleds_height;
extern int metaleds_size;

extern int metaleds_led_intensity[16];
extern int metaleds_intensity_led[256];

typedef unsigned char *metaleds_frame_p;

extern int metaleds_init(const char *device_desc);
extern metaleds_frame_p metaleds_malloc();
extern void metaleds_clrsrc(metaleds_frame_p frame);
extern int metaleds_frame(metaleds_frame_p frame);
extern void metaleds_free(metaleds_frame_p frame);

// #define METALEDS_X_Y_TO_IDX(_x, _y) (8*((metaleds_width-1)-_x) + (_y%8) + (_y/8)*8*8*9)
// 
// #define METALEDS_IDX_TO_X(_idx) ((metaleds_width-1) - _idx%(8*8*9)/8)
// #define METALEDS_IDX_TO_Y(_idx) ((_idx/(8*8*9)*8 + (_idx%(8*8*9))%8))

#define METALEDS_X_Y_TO_IDX(_x, _y) ((_x) + (_y)*metaleds_width)

#define METALEDS_IDX_TO_X(_idx) ((_idx) % metaleds_width)
#define METALEDS_IDX_TO_Y(_idx) ((_idx) / metaleds_width)

static inline int metaleds_setpixel(metaleds_frame_p frame, int x, int y, unsigned char intensity)
{
	if (x<0 || x>=metaleds_width || y<0 || y>=metaleds_height)
		return -1;
	frame[METALEDS_X_Y_TO_IDX(x, y)] = metaleds_intensity_led[intensity];
	return 0;
}

#endif

