#include <stdint.h>
#include <avr/io.h>

void irsend_on()
{
	DDRD |= _BV(6);
	TCCR2A = _BV(COM2B0) | _BV(WGM21);
	TCCR2B = _BV(CS20);
	OCR2A = 210;
	OCR2B = 0;
	TIMSK2 = 0;
	TIFR2 = 0;
}

void irsend_off()
{
	DDRD |= _BV(3);
	PORTD &= ~_BV(3);

	TCCR2A = 0;
	TCCR2B = 0;
}

