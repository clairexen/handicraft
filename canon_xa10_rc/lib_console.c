#include <util/delay.h>
#include <inttypes.h>
#include <avr/io.h>
#include <avr/pgmspace.h>
#include <avr/interrupt.h>
#include <avr/wdt.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdbool.h>

void consoleInit(uint32_t baud)
{
	UCSR0A = 0;
	UCSR0B = _BV(RXEN0) | _BV(TXEN0);
	UCSR0C = _BV(UCSZ00) | _BV(UCSZ01);
	UBRR0L = (F_CPU / (baud * 16L) - 1);
	UBRR0H = (F_CPU / (baud * 16L) - 1) >> 8;
	
	DDRD &= ~_BV(PIND0);
	PORTD |= _BV(PIND0);
}

bool consoleAvailable()
{
	return (UCSR0A & _BV(RXC0)) != 0;
}

char consoleGetChar()
{
	while (!consoleAvailable()) { /* wait */ }
	return UDR0;
}

void consolePutChar(char ch)
{
	while ((UCSR0A & _BV(UDRE0)) == 0) { /* wait */ }
	UDR0 = ch;
}

void consolePrint(const char *string)
{
	while (*string)
		consolePutChar(*(string++));
}

void consolePrintDigit(uint8_t digit)
{
	if (digit < 10)
		consolePutChar('0' + digit);
	else
		consolePutChar('A' + digit - 10);
}

void consolePrintNumber(int number, int len, uint8_t base)
{
	uint8_t is_negative = 0;
	uint8_t buffer[6];
	uint8_t *p = buffer;

	if (number < 0) {
		is_negative = 1;
		number *= -1;
		len--;
	}

	do {
		len--;
		*(p++) = number % base;
		number /= base;
	} while (number != 0);

	for (; len > 0; len--)
		consolePutChar(' ');

	if (is_negative)
		consolePutChar('-');
	
	do
		consolePrintDigit(*(--p));
	while (p != buffer);
}

void consolePrintf(const char *fmt, ...)
{
	int i, len;
	char *s;
	va_list ap;
	va_start(ap, fmt);
	while (fmt && *fmt)
	{
		if (*fmt != '%' || *(fmt+1) == 0) {
			consolePutChar(*(fmt++));
			continue;
		}

		fmt++;
		len = 0;
		while ('0' <= *fmt && *fmt <= '9')
			len = len*10 + (*(fmt++) - '0');
		switch (*(fmt++))
		{
		case 's':
			s = va_arg(ap, char*);
			if (len != 0) {
				char *p = s;
				while (*(p++) != 0)
					len--;
				while (len-- > 0)
					consolePutChar(' ');
			}
			consolePrint(s);
			break;
		case 'c':
			i = va_arg(ap, int);
			consolePutChar(i);
			break;
		case 'x':
			i = va_arg(ap, int);
			consolePrintNumber(i, len, 16);
			break;
		case 'd':
			i = va_arg(ap, int);
			consolePrintNumber(i, len, 10);
			break;
		case 'o':
			i = va_arg(ap, int);
			consolePrintNumber(i, len, 8);
			break;
		case 'b':
			i = va_arg(ap, int);
			consolePrintNumber(i, len, 2);
			break;
		case '%':
			consolePutChar('%');
			break;
		default:
			consolePutChar(*(fmt-2));
			consolePutChar(*(fmt-1));
			break;
		}
	}
	va_end(ap);
}

int consoleReadDecimal(const char *prompt)
{
	int v = 0, sign = 1, ch;

	consolePrint(prompt);

	do {
		ch = consoleGetChar();
		if (ch == '-') {
			sign = -sign;
			consolePutChar(ch);
		}
		if (ch == '+') {
			sign = +sign;
			consolePutChar(ch);
		}
	} while (ch < '0' || ch > '9');

	while (ch >= '0' && ch <= '9') {
		v = v * 10 + (ch - '0');
		consolePutChar(ch);
		ch = consoleGetChar();
	}

	consolePutChar('\r');
	consolePutChar('\n');
	return v * sign;
}

