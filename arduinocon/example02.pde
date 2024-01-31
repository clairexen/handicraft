
#include <Console.h>

void setup()
{
	consoleInit(9600);
	consolePrint("\n\n");
	consolePrintf("sizeof(short) ....... %d\n", sizeof(short));
	consolePrintf("sizeof(int) ......... %d\n", sizeof(int));
	consolePrintf("sizeof(long) ........ %d\n", sizeof(long));
	consolePrintf("sizeof(long long) ... %d\n", sizeof(long long));
	consolePrintf("sizeof(void*) ....... %d\n", sizeof(void*));
}

void loop()
{
}

