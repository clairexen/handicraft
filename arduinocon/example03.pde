
#include <Console.h>

const char keyMap[5][4] = { "QAZ", "WSX", "EDC", "RFV", "TGB" };
const char pinId[5] = { 9, 8, 7, 6, 5 };

char pinStatus[5] = { 'Z', 'Z', 'Z', 'Z', 'Z' };

void menu()
{
	consolePrint("\n");
	consolePrint("           +-------+-------+-------+-------+-------+\n");
	consolePrint("           | Pin 9 | Pin 8 | Pin 7 | Pin 6 | Pin 5 |\n");
	consolePrint("           +-------+-------+-------+-------+-------+\n");
	consolePrint(" Set HIGH: |   Q   |   W   |   E   |   R   |   T   |\n");
	consolePrint("           +-------+-------+-------+-------+-------+\n");
	consolePrint(" Set Z:    |   A   |   S   |   D   |   F   |   G   |\n");
	consolePrint("           +-------+-------+-------+-------+-------+\n");
	consolePrint(" Set LOW:  |  Y/Z  |   X   |   C   |   V   |   B   |\n");
	consolePrint("           +-------+-------+-------+-------+-------+\n");
	consolePrint("\n");
}

void update()
{
	for (int i=0; i<5; i++) {
		if (i)
			consolePrint(", ");
		if (pinStatus[i] == 'H') {
			pinMode(pinId[i], OUTPUT);
			digitalWrite(pinId[i], HIGH);
			consolePrintf("p%d=H", pinId[i]);
		}
		if (pinStatus[i] == 'Z') {
			pinMode(pinId[i], INPUT);
			digitalWrite(pinId[i], LOW);
			consolePrintf("p%d=Z", pinId[i]);
		}
		if (pinStatus[i] == 'L') {
			pinMode(pinId[i], OUTPUT);
			digitalWrite(pinId[i], LOW);
			consolePrintf("p%d=L", pinId[i]);
		}
	}
	consolePrint("> ");
}

void setup()
{
	consoleInit(9600);
	consolePrint("Starting up...\n");

	menu();
	update();
}

void loop()
{
	if (consoleAvailable()) {
		boolean handled = false;
		char ch = consoleGetChar();
		if (ch >= 'a' && ch <= 'z')
			ch -= 'a' - 'A';
		if (ch == 'Y')
			ch = 'Z';
		for (int i=0; i<5; i++) {
			if (ch == keyMap[i][0])
				handled = true, pinStatus[i] = 'H';
			if (ch == keyMap[i][1])
				handled = true, pinStatus[i] = 'Z';
			if (ch == keyMap[i][2])
				handled = true, pinStatus[i] = 'L';
		}
		if (!handled) {
			consolePrint("?\n");
			menu();
		} else {
			consolePrintf("%c\n", ch);
		}
		update();
	}
}

