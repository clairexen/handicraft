#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

uint32_t title[] = {
	0b00001110111010100010111011100000,
	0b00001010101010110110100010000000,
	0b00001110111010101010110011100000,
	0b00001000110010100010100000100000,
	0b00001000101010100010111011100000
};

void setpixel(int x, int y, uint8_t r, uint8_t g, uint8_t b)
{
	if (0 <= x && x < 32 && 0 <= y && y < 32) {
		uint32_t rgb = (r << 16) | (g << 8) | b;
		uint32_t addr = 4*x + 32*4*y + 0x10000000;
		*(volatile uint32_t*)addr = rgb;
	}
}

#define MAX_PRIMES 1000000
unsigned char p[MAX_PRIMES/16+1]; /* zero-initialized */

int primes_lt_100 = 0;
int primes_lt_1000 = 0;
int primes_lt_10000 = 0;
int primes_lt_100000 = 0;
int primes_lt_1000000 = 0;

void update_progress(int p)
{
	static int progress_x = 5;
	static int progress_n = 0;

	while (progress_n < primes_lt_1000000)
	{
		for (int y = 14; y < 18; y++)
			setpixel(progress_x, y, 255, 0, 0);
		
		progress_x++;
		progress_n += 3569;
	}

	for (int x = 26; x >= 5; x--)
	{
		if (p & 1) {
			setpixel(x, 24, 255, 255, 0);
			setpixel(x, 25, 255, 255, 0);
			setpixel(x, 26, 255, 255, 0);
		} else {
			setpixel(x, 24, 8, 8, 8);
			setpixel(x, 25, 8, 8, 8);
			setpixel(x, 26, 8, 8, 8);
		}

		p = p >> 1;
	}
}

void print_primes()
{
	int a, b, k = 1;

	printf("%6d", 2);
	for (a = 3; a < MAX_PRIMES; a += 2)
	{
		if (p[a >> 4] & (1 << ((a >> 1) & 7)))
			continue;

		for (b = a + a; b < MAX_PRIMES; b += a)
			if (b & 1)
				p[b >> 4] = p[b >> 4] | (1 << ((b >> 1) & 7));

		k++;
		if (a < 100) primes_lt_100 = k;
		if (a < 1000) primes_lt_1000 = k;
		if (a < 10000) primes_lt_10000 = k;
		if (a < 100000) primes_lt_100000 = k;
		if (a < 1000000) primes_lt_1000000 = k;

		update_progress(a);
		printf("%c%6d", (k-1) % 8 == 0 ? '\n' : ' ', a);
		fflush(stdout);
	}

	printf("\n");
	printf("#primes <     100: %5d\n", primes_lt_100);
	printf("#primes <    1000: %5d\n", primes_lt_1000);
	printf("#primes <   10000: %5d\n", primes_lt_10000);
	printf("#primes <  100000: %5d\n", primes_lt_100000);
	printf("#primes < 1000000: %5d\n", primes_lt_1000000);

	bool got_error = false;
	if (primes_lt_100     !=    25) got_error = true;
	if (primes_lt_1000    !=   168) got_error = true;
	if (primes_lt_10000   !=  1229) got_error = true;
	if (primes_lt_100000  !=  9592) got_error = true;
	if (primes_lt_1000000 != 78498) got_error = true;

	printf(got_error ? "ERROR!!!\n" : "OK.\n");

	putchar(0);
	fflush(stdout);
}

void main()
{
	for (int x = 0; x < 32; x++)
	for (int y = 0; y < 32; y++)
		setpixel(x, y, 0, 0, 0);

	for (int x = 0; x < 32; x++)
	for (int y = 0; y < 5; y++)
		if ((title[y] >> (31-x)) & 1)
			setpixel(x, y+3, 255, 255, 255);

	for (int x = 4; x < 28; x++)
	for (int y = 13; y < 19; y++)
		setpixel(x, y, 255, 255, 255);

	for (int x = 5; x < 27; x++)
	for (int y = 14; y < 18; y++)
		setpixel(x, y, 0, 0, 0);

	print_primes();

	for (int x = 5; x < 27; x++)
	for (int y = 14; y < 18; y++)
		setpixel(x, y, 0, 255, 0);

	asm volatile ("sbreak");
}
