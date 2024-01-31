#include <stdint.h>
#include <stdbool.h>

#define REG_CONSOLE(_addr) (0x30000000 | (_addr))
#define REG_SDCARD(_addr) (0x20000400 | (_addr))

static uint32_t regrd(uint32_t addr)
{
	return *(volatile uint32_t*)addr;
}

static void regwr(uint32_t addr, uint32_t data)
{
	*(volatile uint32_t*)addr = data;
}

// ========================================================================

static void console_putc(int c)
{
	regwr(REG_CONSOLE(0), c);
}

static void console_puth32(uint32_t v)
{
	for (int i = 0; i < 8; i++) {
		int d = v >> 28;
		console_putc(d < 10 ? '0' + d : 'a' + d - 10);
		v = v << 4;
	}
}

static void console_puth8(uint8_t v)
{
	v &= 0xff;
	int d = v >> 4;
	v &= 0x0f;

	console_putc(d < 10 ? '0' + d : 'a' + d - 10);
	console_putc(v < 10 ? '0' + v : 'a' + v - 10);
}

static void console_puts(const char *s)
{
	while (*s)
		regwr(REG_CONSOLE(0), *(s++));
}

// ========================================================================

bool sdcard_ccs_mode;

static void sdcard_cs(bool enable)
{
	if (enable) {
		// console_puts("SD Card EN=1\n");
		regwr(REG_SDCARD(0x0004), 0);
	} else {
		// console_puts("SD Card EN=0\n");
		regwr(REG_SDCARD(0x0004), 1);
	}
}

static uint8_t sdcard_xfer(uint8_t value)
{
	regwr(REG_SDCARD(0x0008), value);
	uint8_t ret = regrd(REG_SDCARD(0x0008));

	// console_puts("SD Card Xfer: ");
	// console_puth8(value);
	// console_putc(' ');
	// console_puth8(ret);
	// console_putc('\n');

	return ret;
}

static const uint8_t sdcard_crc7_table[256] = {
	0x00, 0x12, 0x24, 0x36, 0x48, 0x5a, 0x6c, 0x7e,
	0x90, 0x82, 0xb4, 0xa6, 0xd8, 0xca, 0xfc, 0xee,
	0x32, 0x20, 0x16, 0x04, 0x7a, 0x68, 0x5e, 0x4c,
	0xa2, 0xb0, 0x86, 0x94, 0xea, 0xf8, 0xce, 0xdc,
	0x64, 0x76, 0x40, 0x52, 0x2c, 0x3e, 0x08, 0x1a,
	0xf4, 0xe6, 0xd0, 0xc2, 0xbc, 0xae, 0x98, 0x8a,
	0x56, 0x44, 0x72, 0x60, 0x1e, 0x0c, 0x3a, 0x28,
	0xc6, 0xd4, 0xe2, 0xf0, 0x8e, 0x9c, 0xaa, 0xb8,
	0xc8, 0xda, 0xec, 0xfe, 0x80, 0x92, 0xa4, 0xb6,
	0x58, 0x4a, 0x7c, 0x6e, 0x10, 0x02, 0x34, 0x26,
	0xfa, 0xe8, 0xde, 0xcc, 0xb2, 0xa0, 0x96, 0x84,
	0x6a, 0x78, 0x4e, 0x5c, 0x22, 0x30, 0x06, 0x14,
	0xac, 0xbe, 0x88, 0x9a, 0xe4, 0xf6, 0xc0, 0xd2,
	0x3c, 0x2e, 0x18, 0x0a, 0x74, 0x66, 0x50, 0x42,
	0x9e, 0x8c, 0xba, 0xa8, 0xd6, 0xc4, 0xf2, 0xe0,
	0x0e, 0x1c, 0x2a, 0x38, 0x46, 0x54, 0x62, 0x70,
	0x82, 0x90, 0xa6, 0xb4, 0xca, 0xd8, 0xee, 0xfc,
	0x12, 0x00, 0x36, 0x24, 0x5a, 0x48, 0x7e, 0x6c,
	0xb0, 0xa2, 0x94, 0x86, 0xf8, 0xea, 0xdc, 0xce,
	0x20, 0x32, 0x04, 0x16, 0x68, 0x7a, 0x4c, 0x5e,
	0xe6, 0xf4, 0xc2, 0xd0, 0xae, 0xbc, 0x8a, 0x98,
	0x76, 0x64, 0x52, 0x40, 0x3e, 0x2c, 0x1a, 0x08,
	0xd4, 0xc6, 0xf0, 0xe2, 0x9c, 0x8e, 0xb8, 0xaa,
	0x44, 0x56, 0x60, 0x72, 0x0c, 0x1e, 0x28, 0x3a,
	0x4a, 0x58, 0x6e, 0x7c, 0x02, 0x10, 0x26, 0x34,
	0xda, 0xc8, 0xfe, 0xec, 0x92, 0x80, 0xb6, 0xa4,
	0x78, 0x6a, 0x5c, 0x4e, 0x30, 0x22, 0x14, 0x06,
	0xe8, 0xfa, 0xcc, 0xde, 0xa0, 0xb2, 0x84, 0x96,
	0x2e, 0x3c, 0x0a, 0x18, 0x66, 0x74, 0x42, 0x50,
	0xbe, 0xac, 0x9a, 0x88, 0xf6, 0xe4, 0xd2, 0xc0,
	0x1c, 0x0e, 0x38, 0x2a, 0x54, 0x46, 0x70, 0x62,
	0x8c, 0x9e, 0xa8, 0xba, 0xc4, 0xd6, 0xe0, 0xf2
};

static uint8_t sdcard_crc7(uint8_t crc, uint8_t data)
{
	return sdcard_crc7_table[crc ^ data];
}

static uint16_t sdcard_crc16(uint16_t crc, uint8_t data)
{
	uint16_t x = (crc >> 8) ^ data;
	x ^= x >> 4;
	return (crc << 8) ^ (x << 12) ^ (x << 5) ^ x;
}

static uint8_t sdcard_cmd_r1(uint8_t cmd, uint32_t arg)
{
	uint8_t r1;

	sdcard_cs(true);

	sdcard_xfer(0x40 | cmd);
	sdcard_xfer(arg >> 24);
	sdcard_xfer(arg >> 16);
	sdcard_xfer(arg >> 8);
	sdcard_xfer(arg);

	uint8_t crc = 0;
	crc = sdcard_crc7(crc, 0x40 | cmd);
	crc = sdcard_crc7(crc, arg >> 24);
	crc = sdcard_crc7(crc, arg >> 16);
	crc = sdcard_crc7(crc, arg >> 8);
	crc = sdcard_crc7(crc, arg);
	sdcard_xfer(crc | 1);

	do {
		r1 = sdcard_xfer(0xff);
	} while (r1 == 0xff);

	sdcard_cs(false);
	return r1;
}

static uint8_t sdcard_cmd_rw(uint8_t cmd, uint32_t arg)
{
	uint8_t r1;

	sdcard_cs(true);

	sdcard_xfer(0x40 | cmd);
	sdcard_xfer(arg >> 24);
	sdcard_xfer(arg >> 16);
	sdcard_xfer(arg >> 8);
	sdcard_xfer(arg);

	uint8_t crc = 0;
	crc = sdcard_crc7(crc, 0x40 | cmd);
	crc = sdcard_crc7(crc, arg >> 24);
	crc = sdcard_crc7(crc, arg >> 16);
	crc = sdcard_crc7(crc, arg >> 8);
	crc = sdcard_crc7(crc, arg);
	sdcard_xfer(crc | 1);

	do {
		r1 = sdcard_xfer(0xff);
	} while (r1 == 0xff);

	return r1;
}

static uint8_t sdcard_cmd_r37(uint8_t cmd, uint32_t arg, uint32_t *r37)
{
	uint8_t r1;

	sdcard_cs(true);

	sdcard_xfer(0x40 | cmd);
	sdcard_xfer(arg >> 24);
	sdcard_xfer(arg >> 16);
	sdcard_xfer(arg >> 8);
	sdcard_xfer(arg);

	uint8_t crc = 0;
	crc = sdcard_crc7(crc, 0x40 | cmd);
	crc = sdcard_crc7(crc, arg >> 24);
	crc = sdcard_crc7(crc, arg >> 16);
	crc = sdcard_crc7(crc, arg >> 8);
	crc = sdcard_crc7(crc, arg);
	sdcard_xfer(crc | 1);

	do {
		r1 = sdcard_xfer(0xff);
	} while (r1 == 0xff);

	for (int i = 0; i < 4; i++)
		*r37 = (*r37 << 8) | sdcard_xfer(0xff);

	sdcard_cs(false);
	return r1;
}

static void sdcard_init()
{
	uint8_t r1;
	uint32_t r37;

	regwr(REG_SDCARD(0x0004), 1);
	regwr(REG_SDCARD(0x0000), 20);

	for (int i = 0; i < 10; i++)
		regwr(REG_SDCARD(0x0008), 0xff);

	r1 = sdcard_cmd_r1(0, 0);

	if (r1 != 0x01) {
		console_puts("Unexpected SD Card CMD0 R1: ");
		console_puth8(r1);
		console_putc('\n');
		while (1) { }
	}

	r1 = sdcard_cmd_r1(59, 1);

	if (r1 != 0x01) {
		console_puts("Unexpected SD Card CMD59 R1: ");
		console_puth8(r1);
		console_putc('\n');
		while (1) { }
	}

	r1 = sdcard_cmd_r37(8, 0x1ab, &r37);
	if (r1 != 0x01 || (r37 & 0xfff) != 0x1ab) {
		console_puts("Unexpected SD Card CMD8 R1 / R7: ");
		console_puth8(r1);
		console_putc(' ');
		console_puth32(r37);
		console_putc('\n');
		while (1) { }
	}

	r1 = sdcard_cmd_r37(58, 0, &r37);

	if (r1 != 0x01) {
		console_puts("Unexpected SD Card CMD58 R1: ");
		console_puth8(r1);
		console_putc('\n');
		while (1) { }
	}

	if ((r37 & 0x00300000) == 0) {
		console_puts("SD Card doesn't support 3.3V! OCR reg: ");
		console_puth32(r37);
		console_putc('\n');
		while (1) { }
	}

	for (int i = 0;; i++)
	{
		// ACMD41, set HCS
		sdcard_cmd_r1(55, 0);
		r1 = sdcard_cmd_r1(41, 1 << 30);

		if (r1 == 0x00)
			break;

		if (r1 != 0x01) {
			console_puts("Unexpected SD Card ACMD41 R1: ");
			console_puth8(r1);
			console_putc('\n');
			while (1) { }
		}

		if (i == 10000) {
			console_puts("Timeout on SD Card ACMD41.\n");
			while (1) { }
		}
	}

	r1 = sdcard_cmd_r37(58, 0, &r37);

	if (r1 != 0x00) {
		console_puts("Unexpected SD Card CMD58 R1: ");
		console_puth8(r1);
		console_putc('\n');
		while (1) { }
	}

	sdcard_ccs_mode = !!(r37 & (1 << 30));

	r1 = sdcard_cmd_r1(16, 512);

	if (r1 != 0x00) {
		console_puts("Unexpected SD Card CMD16 R1: ");
		console_puth8(r1);
		console_putc('\n');
		while (1) { }
	}
}

static void sdcard_read(uint8_t *data, uint32_t blockaddr)
{
	if (!sdcard_ccs_mode)
		blockaddr <<= 9;

	uint8_t r1 = sdcard_cmd_rw(17, blockaddr);

	if (r1 != 0x00) {
		console_puts("Unexpected SD Card CMD17 R1: ");
		console_puth8(r1);
		console_putc('\n');
		while (1) { }
	}

	while (1) {
		r1 = sdcard_xfer(0xff);
		if (r1 == 0xfe) break;
		if (r1 == 0xff) continue;
		console_puts("Unexpected SD Card CMD17 data token: ");
		console_puth8(r1);
		console_putc('\n');
		while (1) { }
	}

	uint16_t crc = 0x0;
	for (int i = 0; i < 512; i++) {
		data[i] = sdcard_xfer(0xff);
		crc = sdcard_crc16(crc, data[i]);
	}

	crc = sdcard_crc16(crc, sdcard_xfer(0xff));
	crc = sdcard_crc16(crc, sdcard_xfer(0xff));

	if (crc != 0) {
		console_puts("CRC Error while reading from SD Card!\n");
		while (1) { }
	}

	sdcard_cs(false);
}

static void sdcard_write(const uint8_t *data, uint32_t blockaddr)
{
	if (!sdcard_ccs_mode)
		blockaddr <<= 9;

	uint8_t r1 = sdcard_cmd_rw(24, blockaddr);

	if (r1 != 0x00) {
		console_puts("Unexpected SD Card CMD17 R1: ");
		console_puth8(r1);
		console_putc('\n');
		while (1) { }
	}

	sdcard_xfer(0xff);
	sdcard_xfer(0xfe);

	uint16_t crc = 0x0;
	for (int i = 0; i < 512; i++) {
		crc = sdcard_crc16(crc, data[i]);
		sdcard_xfer(data[i]);
	}

	sdcard_xfer(crc >> 8);
	sdcard_xfer(crc);
	sdcard_xfer(0xff);

	sdcard_cs(false);
}

// ========================================================================

int main()
{
	sdcard_init();
	console_puts("SD Card Initialized.\n");

	uint8_t buffer[512];

	sdcard_read(buffer, 0);
	for (int i = 0; i < 512; i++) {
		console_puth8(buffer[i]);
		console_putc(i % 32 == 31 ? '\n' : ' ');
	}
	console_putc('\n');

	sdcard_read(buffer, 1);
	for (int i = 0; i < 512; i++) {
		console_puth8(buffer[i]);
		console_putc(i % 32 == 31 ? '\n' : ' ');
	}
	console_putc('\n');

	// for (int i = 0; i < 512; i++)
	// 	buffer[i] = 0xff;
	// sdcard_write(buffer, 1);

	// sdcard_read(buffer, 0);
	// for (int i = 0; i < 512; i++) {
	// 	console_puth8(buffer[i]);
	// 	console_putc(i % 32 == 31 ? '\n' : ' ');
	// }
	// console_putc('\n');

	// sdcard_read(buffer, 1);
	// for (int i = 0; i < 512; i++) {
	// 	console_puth8(buffer[i]);
	// 	console_putc(i % 32 == 31 ? '\n' : ' ');
	// }
	// console_putc('\n');

	return 0;
}
