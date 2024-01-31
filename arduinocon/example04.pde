
#include <Console.h>

int pin_in = 12;
int pin_out = 13;

int duration_in = 0;
int duration_out = 1000;

void setup()
{
	consoleInit(9600);
	consolePrint("Starting up...\n");
	consolePrintf("Reading PWM from pin %d and writing PWM to pin %d.\n", pin_in, pin_out);
	consolePrint("Output PWM may be controlled using keys 0 .. 9.\n");

	pinMode(pin_in, INPUT);
	pinMode(pin_out, OUTPUT);
}

void loop()
{
	if (digitalRead(pin_in) != HIGH) {
		int duration = pulseIn(pin_in, HIGH, 10000);
		if (duration > 0)
			duration_in = duration;
	}

	if (consoleAvailable()) {
		char ch = consoleGetChar();
		if (ch >= '0' && ch <= '9')
			duration_out = 800 + (ch-'0')*100;
	}

	digitalWrite(pin_out, HIGH);
	delayMicroseconds(duration_out);
	digitalWrite(pin_out, LOW);

	consolePrintf("In:%d Out:%d   \r", duration_in, duration_out);
}

