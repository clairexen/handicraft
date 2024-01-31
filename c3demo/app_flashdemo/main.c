#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

volatile uint32_t *const flashreg = (void*)0x20000050;

void spi_flash_cs(bool enable)
{
	if (enable)
		*flashreg &= ~8;
	else
		*flashreg |= 8;
}

uint8_t spi_flash_xfer_bit(uint8_t value)
{
	// set or clear MOSI, clock down
	if (value != 0)
		*flashreg = (*flashreg | 2) & ~4;
	else
		*flashreg = (*flashreg & ~2) & ~4;
	
	// sample out bit, clock up
	uint32_t r = *flashreg;
	*flashreg = r | 4;

	return r & 1;
}

uint8_t spi_flash_xfer_byte(uint8_t value)
{
	uint8_t outval = 0, bitmask = 0x80;

	while (bitmask != 0) {
		if (spi_flash_xfer_bit(value & bitmask))
			outval |= bitmask;
		bitmask = bitmask >> 1;
	}

	return outval;
}

void flash_power_up()
{
	spi_flash_cs(true);
	spi_flash_xfer_byte(0xab);
	spi_flash_cs(false);
}

void flash_power_down()
{
	spi_flash_cs(true);
	spi_flash_xfer_byte(0xb9);
	spi_flash_cs(false);
}

void flash_read_id()
{
	printf("Reading flash ID:");

	spi_flash_cs(true);
	spi_flash_xfer_byte(0x9f);

	for (int i = 0; i < 20; i++)
		printf(" %02x", spi_flash_xfer_byte(0));

	spi_flash_cs(false);
	printf("\n");
}

void flash_read_data()
{
	printf("Reading first 256 bytes of flash data:\n");

	spi_flash_cs(true);
	spi_flash_xfer_byte(0x03);
	spi_flash_xfer_byte(0);
	spi_flash_xfer_byte(0);
	spi_flash_xfer_byte(0);

	for (int i = 0; i < 256; i += 16)
	{
		printf("%04x:", i);

		char ascii_buf[17];
		ascii_buf[16] = 0;

		for (int k = 0; k < 16; k++) {
			uint8_t v = spi_flash_xfer_byte(0);
			ascii_buf[k] = (32 <= v && v < 128) ? v : '.';
			printf("%s%02x", k % 4 == 0 ? "  " : " ", v);
		}

		printf("  |%s|\n", ascii_buf);
	}

	spi_flash_cs(false);
}

void main()
{
	spi_flash_cs(false);
	flash_power_up();
	flash_read_id();
	flash_read_data();
	flash_power_down();
	putchar(0);
	fflush(stdout);
	asm volatile ("sbreak");
}

