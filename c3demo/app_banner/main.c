#include <stdint.h>
#include <stdbool.h>
#include "8x8font.h"

static inline void setled1(bool v)
{
	if (v)
		*(volatile uint32_t*)0x20000000 |= 1;
	else
		*(volatile uint32_t*)0x20000000 &= ~1;
}

static inline void setled2(bool v)
{
	if (v)
		*(volatile uint32_t*)0x20000000 |= 2;
	else
		*(volatile uint32_t*)0x20000000 &= ~2;
}

static inline void setled3(bool v)
{
	if (v)
		*(volatile uint32_t*)0x20000000 |= 4;
	else
		*(volatile uint32_t*)0x20000000 &= ~4;
}

static inline void setdebug0(bool v)
{
	if (v)
		*(volatile uint32_t*)0x20000000 |= 8;
	else
		*(volatile uint32_t*)0x20000000 &= ~8;
}

static inline void setdebug1(bool v)
{
	if (v)
		*(volatile uint32_t*)0x20000000 |= 16;
	else
		*(volatile uint32_t*)0x20000000 &= ~16;
}

int console_peek_ch = -1;

static inline int console_peek()
{
	if (console_peek_ch < 0)
		console_peek_ch = *(volatile uint32_t*)0x30000000;
	return console_peek_ch;
}

static inline int console_getc()
{
	int c = console_peek();
	while (c < 0)
		c = console_peek();
	console_peek_ch = -1;
	return c;
}

static inline void console_putc(int c)
{
	*(volatile uint32_t*)0x30000000 = c;
}

static inline void console_puts(const char *s)
{
	while (*s)
		*(volatile uint32_t*)0x30000000 = *(s++);
}

void setpixel(int x, int y, uint8_t r, uint8_t g, uint8_t b)
{
	if (0 <= x && x < 32 && 0 <= y && y < 32) {
		uint32_t rgb = (r << 16) | (g << 8) | b;
		uint32_t addr = 4*x + 32*4*y + 0x10000000;
		*(volatile uint32_t*)addr = rgb;
	}
}

void drawtext(int *xp, int *pp, int y, const char *str)
{
	int p = *pp, x = *xp;

	while (1)
	{
		if (!str[p])
			p = 0;

		uint8_t *f = fontmem + 8*(str[p]);
		uint8_t fl = fontleft[str[p]];
		uint8_t fr = fontright[str[p]];

		if (x < -8) {
			p += 1;
			x += fr-fl+1;
			*pp = p;
			*xp = x;
			continue;
		}

		if (x > 32)
			break;

		for (int oy = 0; oy < 8; oy++, f++) {
			uint8_t b = *f << fl;
			for (int ox = 0; ox <= fr-fl; ox++, b = b << 1)
				if ((b&128) != 0)
					setpixel(x+ox, y+oy, 0, 0, 255);
				else
					setpixel(x+ox, y+oy, 6*(x+ox+1), 6*(y+oy+1), 0);
		}

		p += 1;
		x += fr-fl+1;
	}
}

char top_str[512] = "Yosys ** IceStorm ** Arachne-pnr ** RISC-V ** PicoRV32 ** IcoBoard ** ";
char bottom_str[512] = "Meet us at 32nd Chaos Communication Congress (32c3) - 27-30 December 2015 ++++ ";
bool debug_active = false;

void debug(uint32_t v)
{
	static int x = 0, p = 0;
	static char hexstring[] = "01234567 ";

	for (int i = 0; i < 8; i++)
		hexstring[7-i] = "0123456789abcdef"[(v >> (4*i)) & 15];

	debug_active = true;
	drawtext(&x, &p, 19, hexstring);
	x--;
}

void main()
{
	console_puts("Test application running.\n");

	for (int x = 0; x < 32; x++)
	for (int y = 0; y < 32; y++)
		setpixel(x, y, 6*(x+1), 6*(y+1), 0);

	int x1 = 0, p1 = 0;
	int x2 = 0, p2 = 0;

	for (uint32_t iter = 0; true; iter++)
	{
		uint32_t num_cycles_start;
		asm volatile ("rdcycle %0" : "=r"(num_cycles_start));

		int y1 = debug_active ? 0 : 5, y2 = debug_active ? 8 : 19;

		drawtext(&x1, &p1, y1, top_str);
		drawtext(&x2, &p2, y2, bottom_str);
		x1--, x2--;

		if (iter % 3 == 0) {
			drawtext(&x2, &p2, y2, bottom_str);
			x2--;
		}

		uint32_t num_cycles_now;
		asm volatile ("rdcycle %0" : "=r"(num_cycles_now));

		uint32_t cycles = num_cycles_now - num_cycles_start;
		uint32_t cycles_limit = 0x40000;

		setled1(3*cycles > cycles_limit);
		setled2(3*cycles > 2*cycles_limit);
		setled3(cycles > cycles_limit);
		// debug(cycles);

		while (cycles < cycles_limit) {
			asm volatile ("rdcycle %0" : "=r"(num_cycles_now));
			cycles = num_cycles_now - num_cycles_start;
		}

		if (console_peek() >= 0)
		{
			console_puts("New message: ");

			x1 = p1 = 0;
			for (int i = 0;; i++) {
				top_str[i] = console_getc();
				if (top_str[i] == '\r' || top_str[i] == '\n') {
					top_str[i] = ' ';
					top_str[++i] = ' ';
					top_str[++i] = 0;
					break;
				} else {
					console_putc(top_str[i]);
				}
			}

			console_puts("\nOK\n");
		}
	}
}
