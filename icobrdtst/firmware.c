#include <stdint.h>
#include <stdbool.h>

static void console_putc(int c)
{
	*(volatile uint32_t*)0x30000000 = c;
}

static void console_puts(const char *s)
{
	while (*s)
		*(volatile uint32_t*)0x30000000 = *(s++);
}

void main()
{
	for (int i = 1; i <= 4; i++)
		*(volatile uint32_t*)(0x20000000 + i * 0x100) = 0;

	for (uint8_t a = 0;; a++)
	{
		console_puts("\033[2J\033[;f");
		console_putc("/-\\|"[a % 4]);
		console_puts("\n\n");

		for (int i = 1; i <= 4; i++)
		{
			console_puts("PMOD ");
			console_putc('0' + i);
			console_puts(":");

			uint8_t status = *(volatile uint32_t*)(0x20000008 + i * 0x100);

			console_puts("\n V G");
			for (int k = 7; k >= 4; k--)
				console_puts(((status >> k) & 1) ? " 1" : " 0");

			console_puts("\n V G");
			for (int k = 3; k >= 0; k--)
				console_puts(((status >> k) & 1) ? " 1" : " 0");

			console_puts("\n\n");
		}

		for (int i = 0; i < 100000; i++)
			asm volatile ("");
	}
}

