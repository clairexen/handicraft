
#include "Console.h"
#include <WProgram.h>
#include <stdarg.h>

#ifndef consoleSerial
#define consoleSerial Serial
#endif

int consoleInit(int baud)
{
	consoleSerial.begin(baud);
}

int consoleAvailable()
{
	return consoleSerial.available();
}

int consoleGetChar()
{
	while (1) {
		int ch = consoleSerial.read();
		if (ch >= 0)
			return ch;
	}
}

void consolePutChar(int ch)
{
	if (ch == '\n')
		consoleSerial.print('\r', BYTE);
	consoleSerial.print(ch, BYTE);
}

int consolePrint(const char *string)
{
	while (string && *string)
		consolePutChar(*(string++));
}

int consolePrintf(const char *fmt, ...)
{
	int i;
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
		switch (*(fmt++))
		{
		case 's':
			s = va_arg(ap, char*);
			consolePrint(s);
			break;
		case 'c':
			i = va_arg(ap, int);
			consolePutChar(i);
			break;
		case 'x':
			i = va_arg(ap, int);
			consoleSerial.print(i, HEX);
			break;
		case 'd':
			i = va_arg(ap, int);
			consoleSerial.print(i, DEC);
			break;
		case 'b':
			i = va_arg(ap, int);
			consoleSerial.print(i, BIN);
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

