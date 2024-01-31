#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void saturateRgb_float(float &r, float &g, float &b);
void saturateRgb_byte(uint8_t &r, uint8_t &g, uint8_t &b);

int main()
{
	int num_iter = 1000;
	float err_max = 0, err_mean = 0;
	for (int i = 0; i < num_iter; i++)
	{
		float r = drand48(), g = drand48(), b = drand48();
		float rs = r, gs = g, bs = b;
		saturateRgb_float(rs, gs, bs);
		uint8_t rb = 255*r, gb = 255*g, bb = 255*b;
		saturateRgb_byte(rb, gb, bb);
		float err = sqrt(pow(rs - rb / 255.0, 2) + pow(gs - gb / 255.0, 2) + pow(bs - bb / 255.0, 2));
		printf("%.2f %.2f %.2f | %.2f %.2f %.2f | %.2f %.2f %.2f | %f\n",
			r, g, b, rs, gs, bs, rb / 255.0, gb / 255.0, bb / 255.0, err);
		err_max = fmax(err_max, err);
		err_mean += err / num_iter;
	}
	printf("ERR_MAX:  %f (%5.2f LSB)\n", err_max, err_max*255);
	printf("ERR_MEAN: %f (%5.2f LSB)\n", err_mean, err_mean*255);
	return 0;
}

