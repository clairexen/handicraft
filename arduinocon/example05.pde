
#include <Console.h>

char ch;

void setchar()
{
}

void setup()
{
	consoleInit(19200);
	pinMode(2, OUTPUT);
	ch = consoleReadDecimal("\n\nEnter ASCII code of character to print: ");
}

void loop()
{
	if (consoleAvailable())
		ch = consoleReadDecimal("\n\nEnter ASCII code of character to print: ");
	digitalWrite(2, HIGH);
	delayMicroseconds(500);
	consolePutChar(ch);
	delayMicroseconds(500);
	digitalWrite(2, LOW);
	delay(5);
}

