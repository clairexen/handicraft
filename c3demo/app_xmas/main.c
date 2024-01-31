#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

const uint32_t tree_bitmap[] = {
	0b00000000000000000000000000000000,
	0b00000000000000011000000000000000,
	0b00000000000000011000000000000000,
	0b00000000000000111100000000000000,
	0b00000000000001111110000000000000,
	0b00000000000011111111000000000000,
	0b00000000000111111111100000000000,
	0b00000000001111111111110000000000,
	0b00000000011111111111111000000000,
	0b00000000111111111111111100000000,
	0b00000000000011111111000000000000,
	0b00000000000111111111100000000000,
	0b00000000001111111111110000000000,
	0b00000000011111111111111000000000,
	0b00000000111111111111111100000000,
	0b00000001111111111111111110000000,
	0b00000011111111111111111111000000,
	0b00000000011111111111111000000000,
	0b00000000111111111111111100000000,
	0b00000001111111111111111110000000,
	0b00000011111111111111111111000000,
	0b00000111111111111111111111100000,
	0b00001111111111111111111111110000,
	0b00011111111111111111111111111000,
	0b00000111111111111111111111100000,
	0b00001111111111111111111111110000,
	0b00011111111111111111111111111000,
	0b00111111111111111111111111111100,
	0b01111111111111111111111111111110,
	0b11111111111111111111111111111111,
	0b00000000000001111110000000000000,
	0b00000000000001111110000000000000,
};

#define M 6
#define N 100
uint8_t color_maps[M][N+1][3];

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

uint32_t xorshift32()
{
	static uint32_t x32 = 314159265;
	x32 ^= x32 << 13;
	x32 ^= x32 >> 17;
	x32 ^= x32 << 5;
	return x32 & 31;
}

void make_color_map(int m, int r, int g, int b)
{
	for (int i = 0; i <= N; i++)
	{
		color_maps[m][i][0] = r*(N-i) / N;
		color_maps[m][i][1] = (g*(N-i) + 255*i) / N;
		color_maps[m][i][2] = b*(N-i) / N;
	}
}

void main()
{
	for (int x = 0; x < 32; x++)
	for (int y = 0; y < 32; y++)
		setpixel(x, y, 0, 0, 0);

	for (int x = 0; x < 32; x++)
	for (int y = 0; y < 32; y++)
		if ((tree_bitmap[y] >> x) & 1)
			setpixel(x, y, 0, 255, 0);

	make_color_map(0, 255, 0, 0);
	make_color_map(1, 0, 0, 255);
	make_color_map(2, 0, 255, 255);
	make_color_map(3, 255, 0, 255);
	make_color_map(4, 255, 255, 0);
	make_color_map(5, 255, 255, 255);

	int blink_m[N+1] = {};
	int blink_x[N+1] = {};
	int blink_y[N+1] = {};

	while (1)
	{
		for (int k = N; k >= 0; k--)
		{
			if ((tree_bitmap[blink_y[k]] >> blink_x[k]) & 1)
				setpixel(blink_x[k], blink_y[k], color_maps[blink_m[k]][k][0], color_maps[blink_m[k]][k][1], color_maps[blink_m[k]][k][2]);
			if (k == 0) {
				do { blink_m[k] = xorshift32(); } while (blink_m[k] >= M);
				blink_x[k] = xorshift32();
				blink_y[k] = xorshift32();
			} else {
				blink_m[k] = blink_m[k-1];
				blink_x[k] = blink_x[k-1];
				blink_y[k] = blink_y[k-1];
			}
		}

		for (int i = 0; i < 1000; i++)
			asm volatile ("");
	}
}
