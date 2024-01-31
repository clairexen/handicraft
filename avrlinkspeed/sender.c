#include <stdint.h>
#include <stdbool.h>
#include <avr/io.h>
#include <avr/sleep.h>
#include <avr/interrupt.h>

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

static void serio_send(uint8_t dat)
{
	while ((UCSR0A & _BV(UDRE0)) == 0) { }
	UDR0 = dat;
}

static uint8_t serio_recv()
{
	while ((UCSR0A & _BV(RXC0)) == 0) { }
        return UDR0;
}

int main()
{
	uint8_t i = 0;
	serio_setup();
	while (1)
		serio_send(i++);
	return 0;
}

