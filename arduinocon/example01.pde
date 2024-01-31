
#include <Console.h>

void setup()
{
	consoleInit(9600);
	consolePrint("Starting up...\n");
}

void loop()
{
	int a, b;

	consolePrint("\n");

	a = consoleReadDecimal("Enter first number: ");
	b = consoleReadDecimal("Enter second number: ");

	consolePrintf("Sum of both numbers: %d\n", a + b);
}

