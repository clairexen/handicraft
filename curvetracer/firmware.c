
#include <inttypes.h>
#include <avr/io.h>
#include <avr/pgmspace.h>
#include <avr/interrupt.h>
#include <avr/wdt.h>

#define BAUD_RATE 38400

static void uart_setup()
{
        UBRR0L = (uint8_t) (F_CPU / (BAUD_RATE * 16L) - 1);
        UBRR0H = (F_CPU / (BAUD_RATE * 16L) - 1) >> 8;
        UCSR0B = (1 << RXEN0) | (1 << TXEN0);
        UCSR0C = (1 << UCSZ00) | (1 << UCSZ01);

        DDRD &= ~_BV(PIND0);
        PORTD |= _BV(PIND0);
}

static void putch(char ch)
{
        while (!(UCSR0A & _BV(UDRE0))) ;
        UDR0 = ch;
}

static void adc_setup()
{
	// configure mux for channel 0
	ADMUX = _BV(REFS0);

	// configure ADC clock (div = 128)
	ADCSRB = 0;
	ADCSRA = _BV(ADPS2) | _BV(ADPS1) | _BV(ADPS0);

	// start ADC
	ADCSRA |= _BV(ADEN) | _BV(ADSC) | _BV(ADIF);

	// configure mux for channel 1
	ADMUX = _BV(REFS0) | 1;
}

// WARNING: This function assumes that the ADC is currently running the conversion
// for ADC channel 'i' and is already pre-configured for 'i+1'. It starts the 'i+1'
// conversion and pre-configures 'i+2'. You can't call this function with an out-of
// order argument and expect it to work..
static int adc(int i)
{
	int val = 0;

	// wait for adc to be ready
	while ((ADCSRA & _BV(ADIF)) == 0) { }

	// read adc value
	val = ADCL;
	val |= ADCH << 8;

	// start next conversion
	ADCSRA |= _BV(ADSC) | _BV(ADIF);

	// configure mux for next channel
	ADMUX = _BV(REFS0) | (i + 2) % 6;

	return val;
}

int main()
{
	adc_setup();
	uart_setup();
	while (1)
	{
		int v[6], i;
		putch(':');
		for (i = 0; i < 6; i++) {
			v[i] = adc(i);
		}
		for (i = 0; i < 6; i++) {
			putch(' ');
			putch("0123456789abcdef"[(v[i] >> 8) & 15]);
			putch("0123456789abcdef"[(v[i] >> 4) & 15]);
			putch("0123456789abcdef"[(v[i] >> 0) & 15]);
		}
		putch('\r');
		putch('\n');
	}
	return 0;
}

