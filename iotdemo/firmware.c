#include <stdint.h>
#include <stdbool.h>

bool in_flash_mode = false;

static void spi_flash_cs(bool enable)
{
	if (enable)
		*(volatile uint32_t*)0x20000004 &= ~8;
	else
		*(volatile uint32_t*)0x20000004 |= 8;
}

static uint8_t spi_flash_xfer(uint8_t value)
{
	*(volatile uint32_t*)0x20000008 = value;
	return *(volatile uint32_t*)0x20000008;
}

static inline void setled(int v)
{
	*(volatile uint32_t*)0x20000000 = v;
}

static void console_putc(int c)
{
	if (in_flash_mode)
		return;

	*(volatile uint32_t*)0x30000000 = c;
}

static void console_puth32(uint32_t v)
{
	if (in_flash_mode)
		return;

	for (int i = 0; i < 8; i++) {
		int d = v >> 28;
		console_putc(d < 10 ? '0' + d : 'a' + d - 10);
		v = v << 4;
	}
}

static void console_puth8(uint8_t v) __attribute__((unused));

static void console_puth8(uint8_t v)
{
	if (in_flash_mode)
		return;

	v &= 0xff;
	int d = v >> 4;
	v &= 0x0f;

	console_putc(d < 10 ? '0' + d : 'a' + d - 10);
	console_putc(v < 10 ? '0' + v : 'a' + v - 10);
}

static void console_puts(const char *s)
{
	if (in_flash_mode)
		return;

	while (*s)
		*(volatile uint32_t*)0x30000000 = *(s++);
}

static int console_getc()
{
	static int load_timeout = 0;

	while (load_timeout < 1000000)
	{
		int c = *(volatile uint32_t*)0x30000000;
		load_timeout++;

		if (c >= 0) {
			load_timeout = 0;
			return c;
		}
	}

	if (!in_flash_mode)
	{
		console_puts("FLASHBOOT ");
		in_flash_mode = true;
		spi_flash_cs(false);

		// flash_power_up
		spi_flash_cs(true);
		spi_flash_xfer(0xab);
		spi_flash_cs(false);

		// read data at 256 kB offset
		spi_flash_cs(true);
		spi_flash_xfer(0x03);
		spi_flash_xfer(4);
		spi_flash_xfer(0);
		spi_flash_xfer(0);
	}

	int c = spi_flash_xfer(0);

	if (c == '*')
	{
		// end of read
		spi_flash_cs(false);

		// flash_power_down
		spi_flash_cs(true);
		spi_flash_xfer(0xb9);
		spi_flash_cs(false);

		return 0;
	}

	return c;
}

static bool ishex(char ch)
{
	if ('0' <= ch && ch <= '9') return true;
	if ('a' <= ch && ch <= 'f') return true;
	if ('A' <= ch && ch <= 'F') return true;
	return false;
}

static int hex2int(char ch)
{
	if ('0' <= ch && ch <= '9') return ch - '0';
	if ('a' <= ch && ch <= 'f') return ch - 'a' + 10;
	if ('A' <= ch && ch <= 'F') return ch - 'A' + 10;
	return -1;
}

int main()
{
	setled(1); // LEDs ..O

#if 0
	// flash_power_up
	spi_flash_cs(false);
	spi_flash_cs(true);
	spi_flash_xfer(0xab);
	spi_flash_cs(false);

	console_puts("Flash ID:");
	spi_flash_cs(true);
	spi_flash_xfer(0x9f);
	for (int i = 0; i < 20; i++) {
		console_putc(' ');
		console_puth8(spi_flash_xfer(i));
	}
	spi_flash_cs(false);
	console_putc('\n');

	// flash_power_down
	spi_flash_cs(true);
	spi_flash_xfer(0xb9);
	spi_flash_cs(false);
#endif

	console_puts(".\nBootloader> " + 2);
	uint8_t *memcursor = (uint8_t*)(64 * 1024);
	int bytecount = 0;

	while (1)
	{
		setled(2); // LEDs .O.

		char ch = console_getc();

		setled(3); // LEDs .OO

		if (ch == 0 || ch == '@')
		{
			if (bytecount) {
				console_puts("\nWritten 0x");
				console_puth32(bytecount);
				console_puts(" bytes at 0x");
				console_puth32((uint32_t)memcursor);
				console_puts(".\nBootloader> ");
			}

			if (ch == 0) {
				in_flash_mode = false;
				console_puts("RUN\n");
				break;
			}

			int newaddr = 0;
			while (1) {
				ch = console_getc();
				if (!ishex(ch)) break;
				newaddr = (newaddr << 4) | hex2int(ch);
			}

			memcursor = (uint8_t*)newaddr;
			bytecount = 0;
			continue;
		}

		setled(4); // LEDs O..

		if (ishex(ch))
		{
			char ch2 = console_getc();

			if (ishex(ch2)) {
				if (bytecount % 1024 == 0)
					console_putc('.');
				memcursor[bytecount++] = (hex2int(ch) << 4) | hex2int(ch2);
				continue;
			}

			console_putc(ch);
			ch = ch2;
			goto prompt;
		}

		setled(5); // LEDs O.O

		if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n')
			continue;

	prompt:
		setled(6); // LEDs OO.

		console_putc(ch);
		console_puts(".\nBootloader> " + 1);
	}

	setled(0); // LEDs ...
	return 0;
}

