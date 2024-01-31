#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

static inline void setled(int v)
{
	*(volatile uint32_t*)0x20000000 = v;
}

void setpixel(int x, int y, uint8_t r, uint8_t g, uint8_t b)
{
	if (0 <= x && x < 32 && 0 <= y && y < 32) {
		uint32_t rgb = (r << 16) | (g << 8) | b;
		uint32_t addr = 4*x + 32*4*y + 0x10000000;
		*(volatile uint32_t*)addr = rgb;
	}
}

bool getpmod(int pmod_idx, int bit_idx)
{
	uint32_t v = *(volatile uint32_t*)(0x20000004 + 0x10*pmod_idx);
	return (v >> bit_idx) & 1;
}

void main()
{
	for (int x = 0; x < 32; x++)
	for (int y = 0; y < 32; y++)
		setpixel(x, y, 0, 0, 0);

	*(volatile uint32_t*)0x20000010 = 0;
	*(volatile uint32_t*)0x20000020 = 0;
	*(volatile uint32_t*)0x20000030 = 0;
	*(volatile uint32_t*)0x20000040 = 0;

	while (1)
	{
		for (int k = 1; k <= 4; k++)
		for (int i = 0; i < 8; i++)
		{
			if (getpmod(k, i)) {
				setpixel(31 - 4*i - 1, k*6 + 0, 255, 0, 0);
				setpixel(31 - 4*i - 2, k*6 + 0, 255, 0, 0);
				setpixel(31 - 4*i - 1, k*6 + 1, 255, 0, 0);
				setpixel(31 - 4*i - 2, k*6 + 1, 255, 0, 0);
			} else {
				setpixel(31 - 4*i - 1, k*6 + 0, 0, 0, 255);
				setpixel(31 - 4*i - 2, k*6 + 0, 0, 0, 255);
				setpixel(31 - 4*i - 1, k*6 + 1, 0, 0, 255);
				setpixel(31 - 4*i - 2, k*6 + 1, 0, 0, 255);
			}
		}
	}
}
