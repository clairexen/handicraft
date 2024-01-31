#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

uint8_t current_map[32][32];
uint8_t neigh_y_map[32][32];
uint8_t neigh_b_map[32][32];
uint8_t happy_map[32][32];

struct coord_t {
	uint8_t x, y;
};

struct coord_t free_list[32*32];
struct coord_t y_list[32*32];
struct coord_t b_list[32*32];

int free_list_len = 0;
int y_list_len = 0;
int b_list_len = 0;

volatile int current_y1 = 10, current_y2 = 20;
volatile int current_b1 = 10, current_b2 = 20;
volatile int current_speed = 10;

volatile int irq_popdelta_y = 0, irq_popdelta_b = 0;
int popdelta_y = 0, popdelta_b = 0;

volatile extern uint32_t _end;

void free_to_y(int idx)
{
	if (idx < 0 || idx >= free_list_len) {
		printf("Free list idx is out of bounds: %d (len=%d).\n", idx, free_list_len);
		abort();
	}

	int x = free_list[idx].x;
	int y = free_list[idx].y;

	for (int kx = x-1; kx <= x+1; kx++)
	for (int ky = y-1; ky <= y+1; ky++)
		neigh_y_map[kx][ky]++;

	current_map[x][y] = 1;
	y_list[y_list_len++] = free_list[idx];
	free_list[idx] = free_list[--free_list_len];
}

void free_to_b(int idx)
{
	if (idx < 0 || idx >= free_list_len) {
		printf("Free list idx is out of bounds: %d (len=%d).\n", idx, free_list_len);
		abort();
	}

	int x = free_list[idx].x;
	int y = free_list[idx].y;

	for (int kx = x-1; kx <= x+1; kx++)
	for (int ky = y-1; ky <= y+1; ky++)
		neigh_b_map[kx][ky]++;

	current_map[x][y] = 2;
	b_list[b_list_len++] = free_list[idx];
	free_list[idx] = free_list[--free_list_len];
}

void y_to_free(int idx)
{
	if (idx < 0 || idx >= y_list_len) {
		printf("Y list idx is out of bounds: %d (len=%d).\n", idx, y_list_len);
		abort();
	}

	int x = y_list[idx].x;
	int y = y_list[idx].y;

	for (int kx = x-1; kx <= x+1; kx++)
	for (int ky = y-1; ky <= y+1; ky++)
		neigh_y_map[kx][ky]--;

	current_map[x][y] = 0;
	free_list[free_list_len++] = y_list[idx];
	y_list[idx] = y_list[--y_list_len];
}

void b_to_free(int idx)
{
	if (idx < 0 || idx >= b_list_len) {
		printf("B list idx is out of bounds: %d (len=%d).\n", idx, b_list_len);
		abort();
	}

	int x = b_list[idx].x;
	int y = b_list[idx].y;

	for (int kx = x-1; kx <= x+1; kx++)
	for (int ky = y-1; ky <= y+1; ky++)
		neigh_b_map[kx][ky]--;

	current_map[x][y] = 0;
	free_list[free_list_len++] = b_list[idx];
	b_list[idx] = b_list[--b_list_len];
}

static inline void setled(int v)
{
	*(volatile uint32_t*)0x20000000 = v;
}

uint32_t xorshift32(uint32_t n) {
	static uint32_t x32 = 314159265;
	x32 ^= x32 << 13;
	x32 ^= x32 >> 17;
	x32 ^= x32 << 5;
	return x32 % n;
}

void setpixel(int x, int y, uint8_t r, uint8_t g, uint8_t b)
{
	if (0 <= x && x < 32 && 0 <= y && y < 32) {
		uint32_t rgb = (r << 16) | (g << 8) | b;
		uint32_t addr = 4*x + 32*4*y + 0x10000000;
		*(volatile uint32_t*)addr = rgb;
	}
}

void handle_ctrls()
{
	static bool first = true;
	static uint8_t ctrlbits[8];
	bool redraw = first;

	for (int k = 0; k < 8; k++)
	{
		uint8_t bits = ((*(volatile uint32_t*)(0x20000014 + 0x10*(k / 2))) >> ((k % 2) ? 5 : 1)) & 7;

		if (first)
			ctrlbits[k] = bits;

		if (bits == ctrlbits[k])
			continue;

		int old_pos = ctrlbits[k] >> 1;
		int new_pos = bits >> 1;

		old_pos = old_pos ^ (old_pos >> 1);
		new_pos = new_pos ^ (new_pos >> 1);

		int delta = 0;

		if (new_pos == ((old_pos+1) & 3))
			delta = +1;

		if (new_pos == ((old_pos-1) & 3))
			delta = -1;

		if (k == 0 && 0 < current_y1+delta && current_y1+delta < current_y2)
			current_y1 += delta, redraw = true;

		if (k == 5 && current_y1 < current_y2+delta && current_y2+delta < 31)
			current_y2 += delta, redraw = true;

		if (k == 6 && 0 < current_b1+delta && current_b1+delta < current_b2)
			current_b1 += delta, redraw = true;

		if (k == 7 && current_b1 < current_b2+delta && current_b2+delta < 31)
			current_b2 += delta, redraw = true;

		if ((k == 2 || k == 4) && 0 < current_speed-delta && current_speed-delta < 31)
			current_speed -= delta, redraw = true;

		if (k == 1)
			irq_popdelta_y -= 10 * delta;

		if (k == 3)
			irq_popdelta_b -= 10 * delta;

		ctrlbits[k] = bits;
	}

	if (redraw)
	{
		for (int y = 1; y < 31; y++)
		{
			if (y < current_y1)
				setpixel(0, y, 255, 0, 0);
			else if (y <= current_y2)
				setpixel(0, y, 255, 255, 0);
			else
				setpixel(0, y, 255, 0, 0);

			if (y < current_b1)
				setpixel(31, y, 255, 0, 0);
			else if (y <= current_b2)
				setpixel(31, y, 0, 0, 255);
			else
				setpixel(31, y, 255, 0, 0);
		}

		for (int x = 1; x < 31; x++)
			if (x == current_speed)
				setpixel(x, 31, 0, 255, 0);
			else
				setpixel(x, 31, 16, 0, 16);
	}

	first = false;
}

void popupd()
{
	while (popdelta_y < 0) {
		if (y_list_len > 0)
			y_to_free(xorshift32(y_list_len));
		popdelta_y++;
	}

	while (popdelta_b < 0) {
		if (b_list_len > 0)
			b_to_free(xorshift32(b_list_len));
		popdelta_b++;
	}

	while (popdelta_y > 0) {
		if (free_list_len > 0)
			free_to_y(xorshift32(free_list_len));
		popdelta_y--;
	}

	while (popdelta_b > 0) {
		if (free_list_len > 0)
			free_to_b(xorshift32(free_list_len));
		popdelta_b--;
	}
}

void update()
{
	memset(happy_map, 0, 32*32);

	int count_happy_y = 0;
	int count_happy_b = 0;

	popupd();

	for (int idx = 0; idx < y_list_len; idx++)
	{
		int x = y_list[idx].x;
		int y = y_list[idx].y;

		int count_me = neigh_y_map[x][y]-1;
		int count_other = neigh_b_map[x][y];
		int v = 1 + 30*count_other/(count_me+count_other+1);

		if (current_y1 <= v && v < current_y2) {
			happy_map[x][y] = 1;
			count_happy_y++;
		} else
		if (xorshift32(128) < current_speed) {
			y_to_free(idx);
			popdelta_y++;
		}
	}

	for (int idx = 0; idx < b_list_len; idx++)
	{
		int x = b_list[idx].x;
		int y = b_list[idx].y;

		int count_me = neigh_b_map[x][y]-1;
		int count_other = neigh_y_map[x][y];
		int v = 1 + 30*count_other/(count_me+count_other+1);

		if (current_b1 <= v && v < current_b2) {
			happy_map[x][y] = 1;
			count_happy_b++;
		} else
		if (xorshift32(128) < current_speed) {
			b_to_free(idx);
			popdelta_b++;
		}
	}

	popupd();

	int xy = 1;
	for (int i = 0; i < y_list_len; i += 26, xy++) {
		if (i < count_happy_y)
			setpixel(xy, 0, 255, 255, 0);
		else
			setpixel(xy, 0, 16, 16, 0);
	}

	int xb = 30;
	for (int i = 0; i < b_list_len && xb != xy-1; i += 26, xb--) {
		if (i < count_happy_b)
			setpixel(xb, 0, 0, 0, 255);
		else
			setpixel(xb, 0, 0, 0, 16);
	}

	while (xy <= xb)
		setpixel(xy++, 0, 0, 0, 0);
}

void redraw()
{
	for (int x = 2; x < 30; x++)
	for (int y = 2; y < 30; y++) {
		if (current_map[x][y] == 0) {
			setpixel(x, y, 0, 0, 0);
		} else
		if (current_map[x][y] == 1) {
			if (happy_map[x][y])
				setpixel(x, y, 255, 255, 0);
			else
				setpixel(x, y, 16, 16, 0);
		} else
		if (current_map[x][y] == 2) {
			if (happy_map[x][y])
				setpixel(x, y, 0, 0, 255);
			else
				setpixel(x, y, 0, 0, 16);
		}
	}
}

// external symbol
void irq_wrapper();
void reset_timer();
void maskirq();
void unmaskirq();

void install_irq()
{
	uint32_t rel_addr = (uint32_t)irq_wrapper - 0x10;
	uint32_t jal_instr = 0x6f;

	uint32_t rel_addr_20 = (rel_addr >> 20) & 1;
	uint32_t rel_addr_10_1 = (rel_addr >> 1) & 0x3ff;
	uint32_t rel_addr_11 = (rel_addr >> 11) & 1;
	uint32_t rel_addr_19_12 = (rel_addr >> 12) & 0xff;

	jal_instr |= rel_addr_19_12 << 12;
	jal_instr |= rel_addr_11 << (12+8);
	jal_instr |= rel_addr_10_1 << (12+8+1);
	jal_instr |= rel_addr_20 << (12+8+1+10);

	*(uint32_t*)0x00000010 = jal_instr;
}

uint32_t *irq(uint32_t *regs, uint32_t irqs)
{
	handle_ctrls();
	reset_timer();
	return regs;
}

void check()
{
	for (int i = 0; i < free_list_len; i++)
	{
		int x = free_list[i].x;
		int y = free_list[i].y;

		if (current_map[x][y] != 0) {
			printf("Free idx %d: Expected current_map[%d][%d] == 0, but is %d.\n", i, x, y, current_map[x][y]);
			abort();
		}
	}

	for (int i = 0; i < y_list_len; i++)
	{
		int x = y_list[i].x;
		int y = y_list[i].y;

		if (current_map[x][y] != 1) {
			printf("Y idx %d: Expected current_map[%d][%d] == 1, but is %d.\n", i, x, y, current_map[x][y]);
			abort();
		}
	}

	for (int i = 0; i < b_list_len; i++)
	{
		int x = b_list[i].x;
		int y = b_list[i].y;

		if (current_map[x][y] != 2) {
			printf("B idx %d: Expected current_map[%d][%d] == 2, but is %d.\n", i, x, y, current_map[x][y]);
			abort();
		}
	}

	if ((free_list_len+y_list_len+b_list_len) != 28*28 || free_list_len < 0 || y_list_len < 0 || b_list_len < 0) {
		printf("Lists got unbalanced: free=%d, y=%d, b=%d\n", free_list_len, y_list_len, b_list_len);
		abort();
	}

	if (_end != 0xdeadbeef) {
		printf("Magic _end mark got overwritten: %x\n", _end);
		abort();
	}
}

void main()
{
	_end = 0xdeadbeef;

	for (int x = 0; x < 32; x++)
	for (int y = 0; y < 32; y++)
		setpixel(x, y, 0, 0, 0);

	for (int x = 2; x < 30; x++)
	for (int y = 2; y < 30; y++) {
		free_list[free_list_len].x = x;
		free_list[free_list_len++].y = y;
	}

	*(volatile uint32_t*)0x20000010 = 0;
	*(volatile uint32_t*)0x20000020 = 0;
	*(volatile uint32_t*)0x20000030 = 0;
	*(volatile uint32_t*)0x20000040 = 0;

	install_irq();
	reset_timer();

	while (1)
	{
		maskirq();
		popdelta_y += irq_popdelta_y;
		popdelta_b += irq_popdelta_b;
		irq_popdelta_y = 0;
		irq_popdelta_b = 0;
		// printf("%d | %d %d | %d %d\n", free_list_len, y_list_len, b_list_len, popdelta_y, popdelta_b);
		unmaskirq();

		setled(1);
		check();
		update();

		setled(2);
		redraw();
	}
}
