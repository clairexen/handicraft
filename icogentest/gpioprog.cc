#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <wiringPi.h>

#define ICE_CRST  6
#define ICE_MOSI 26
#define ICE_MISO 27
#define ICE_SS_B 28
#define ICE_SCLK 29

int main()
{
	wiringPiSetup();

	pinMode(ICE_CRST, OUTPUT);
	pinMode(ICE_MOSI, OUTPUT);
	pinMode(ICE_MISO, INPUT);
	pinMode(ICE_SS_B, OUTPUT);
	pinMode(ICE_SCLK, OUTPUT);

	printf("reset..\n");

	digitalWrite(ICE_SCLK, HIGH);
	digitalWrite(ICE_SS_B, LOW);
	digitalWrite(ICE_CRST, LOW);
	usleep(100);

	digitalWrite(ICE_CRST, HIGH);
	usleep(2000);

	printf("programming..\n");

	while (1)
	{
		int byte = getchar();
		if (byte < 0)
			break;
		for (int i = 7; i >= 0; i--) {
			digitalWrite(ICE_MOSI, ((byte >> i) & 1) ? HIGH : LOW);
			digitalWrite(ICE_SCLK, LOW);
			digitalWrite(ICE_SCLK, HIGH);
		}
	}

	for (int i = 0; i < 49; i++) {
		digitalWrite(ICE_SCLK, LOW);
		digitalWrite(ICE_SCLK, HIGH);
	}

	printf("done.\n");

	pinMode(ICE_CRST, INPUT);
	pinMode(ICE_MOSI, INPUT);
	pinMode(ICE_MISO, INPUT);
	pinMode(ICE_SS_B, INPUT);
	pinMode(ICE_SCLK, INPUT);

	return 0;
}

