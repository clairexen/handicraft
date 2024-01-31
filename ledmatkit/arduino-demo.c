
// simple 8x8 ledmatrix demo using an arduino-based prototype

/*** build and install instructions ***
avr-gcc -Os -std=gnu99 -mmcu=atmega328p -o arduino-demo.elf arduino-demo.c
avr-objcopy -O ihex -R .eeprom arduino-demo.elf arduino-demo.hex
avrdude -v -pm328p -cstk500v1 -P/dev/ttyUSB2 -b57600 -D -Uflash:w:arduino-demo.hex:i
**************************************/

#include <avr/io.h>

volatile uint8_t *colPinPort[8] = {
	&PORTB, &PORTC, &PORTB, &PORTC, &PORTB, &PORTC, &PORTB, &PORTC
};
uint8_t colPinMask[8] = {
	_BV(5), _BV(0), _BV(4), _BV(1), _BV(3), _BV(2), _BV(2), _BV(3)
};

#define WAITSTATE_MUL 5
uint16_t colorWaitStates[16] = {
	 0 * WAITSTATE_MUL,  10 * WAITSTATE_MUL,  12 * WAITSTATE_MUL,  15 * WAITSTATE_MUL,
	18 * WAITSTATE_MUL,  22 * WAITSTATE_MUL,  27 * WAITSTATE_MUL,  33 * WAITSTATE_MUL,
	39 * WAITSTATE_MUL,  47 * WAITSTATE_MUL,  56 * WAITSTATE_MUL,  68 * WAITSTATE_MUL,
	82 * WAITSTATE_MUL, 100 * WAITSTATE_MUL, 120 * WAITSTATE_MUL, 150 * WAITSTATE_MUL
};

uint8_t fb1[8][8] = {
	{0, 1,  2,  3,  4,  5,  6,  7},
	{0, 1,  2,  3,  4,  5,  6,  7},
	{0, 0,  0,  0,  0,  0,  0,  0},
	{4, 5,  6,  7,  8,  9, 10, 12},
	{4, 5,  6,  7,  8,  9, 10, 12},
	{0, 0,  0,  0,  0,  0,  0,  0},
	{8, 9, 10, 11, 12, 13, 14, 15},
	{8, 9, 10, 11, 12, 13, 14, 15},
};

uint8_t fb2[8][8] = {
	{15, 0, 0, 0, 0, 0, 0, 15},
	{0, 15, 0, 0, 0, 0, 15, 0},
	{0, 0, 15, 0, 0, 15, 0, 0},
	{0, 0, 0, 15, 15, 0, 0, 0},
	{0, 0, 0, 15, 15, 0, 0, 0},
	{0, 0, 15, 0, 0, 15, 0, 0},
	{0, 15, 0, 0, 0, 0, 15, 0},
	{15, 0, 0, 0, 0, 0, 0, 15},
};

uint8_t (*fb)[8][8];

void main()
{
	UCSR0B = 0;

	DDRB = 0xff;
	PORTB = 0x00;
	DDRC = 0xff;
	PORTC = 0x00;
	DDRD = 0xff;
	PORTD = 0x00;

	uint16_t frame = 0;
	while (1) {
		fb = (frame++ & 0x1fff) < 0x1000 ? &fb1 : &fb2;
		for (uint8_t col = 0; col < 8; col++) {
			uint8_t pd_data[16] = { 0, 0, 0, 0, 0, 0, 0, 0 };
			for (uint8_t row = 0; row < 8; row++) {
				uint8_t m = 1 << row, v = (*fb)[row][col];
				for (uint8_t i = 1; i < 16; i++) {
					if (i <= v)
						pd_data[i] |= m;
				}
			}
			*colPinPort[col] |= colPinMask[col];
			uint16_t s = 0;
			for (uint8_t i = 1; i < 16; i++) {
				PORTD = pd_data[i];
				while (s++ < colorWaitStates[i])
					asm volatile ("");
			}
			*colPinPort[col] &= ~colPinMask[col];
		}
	}
}

