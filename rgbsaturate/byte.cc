#include <math.h>
#include <stdint.h>

static uint8_t rgb2h(int r, int g, int b)
{
	int min, max, delta, h;

	min = r < g ? r : g;
	min = min < b ? min : b;

	max = r > g ? r : g;
	max = max > b ? max : b;

	delta = max - min;

	if (max <= 0.0)
		h = 0;
	else if (r >= max)
		h = 32 * (g - b) / delta;
	else if (g >= max)
		h = 64 + 32 * (b - r) / delta;
	else
		h = 128 + 32 * (r - g) / delta;

	if (h < 0)
		h += 192;
	return h;
}

static void h2rgb(uint8_t h, uint8_t &r, uint8_t &g, uint8_t &b)
{
	uint8_t hh = h / 32, ff = (h % 32) << 3;
	switch(hh) {
		case 0:  r = 255;      g = ff;       b = 0;        break;
		case 1:  r = 255 - ff; g = 255;      b = 0;        break;
		case 2:  r = 0;        g = 255;      b = ff;       break;
		case 3:  r = 0;        g = 255 - ff; b = 255;      break;
		case 4:  r = ff;       g = 0;        b = 255;      break;
		default: r = 255;      g = 0;        b = 255 - ff; break;
	}
}

void saturateRgb_byte(uint8_t &r, uint8_t &g, uint8_t &b)
{
	uint8_t h = rgb2h(r, g, b);
	h2rgb(h, r, g, b);
}

