// just trying to saturate a 2MBaud link

#include <avr/io.h>
#include <util/delay.h>

static void serio_setup() {
	// Configure USART0 for 2MBaud
	UBRR0H = 0;
	UBRR0L = 0;
	UCSR0A = _BV(U2X0);
	UCSR0B = _BV(RXEN0) | _BV(TXEN0);
	UCSR0C = _BV(UCSZ00) | _BV(UCSZ01);
	PORTD |= _BV(1);
	DDRD |= _BV(1);
}

int main()
{
	uint8_t counter = 0;
	serio_setup();
	DDRD = 0x04;
	while (1) {
		if ((UCSR0A & _BV(UDRE0)) == 0)
			continue;
		// _delay_us(100);
		UDR0 = counter++;
		PIND = 0x04;
	}
}

